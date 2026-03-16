# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
scheduler_predictor.py
-----------------------

This module provides a small suite of classes to collect runtime
observations from a token‑based scheduler (such as vLLM's unified
scheduler) and to make simple, adaptive decisions about whether new
requests should be admitted into the next scheduling batch.  The goal
is to protect decode latency (TPOT) while still filling the GPU
efficiently.  A sliding window of recent observations is used to
train a lightweight linear regression model that predicts the time
required for a batch given its aggregate token statistics.  When
insufficient data are available the predictor defaults to allowing
all requests so that it can bootstrap its dataset.

The primary entry points are:

* ``SlidingWindowDataCollector`` – collects ``StepObservation``
  instances up to a configurable window size and optionally filters
  observations to only those containing decode tokens.
* ``LinearRegressionModel`` – fits a linear model on collected data
  using ordinary least squares.  It recomputes its coefficients
  whenever new data arrive.
* ``SchedulerPredictor`` – uses the collector and model to decide
  whether admitting an additional request would push per‑token
  latency (TPOT) beyond a target.  When there is too little data the
  predictor conservatively allows all requests in order to collect
  more training examples.

Unit tests demonstrating usage can be found in
``tests/test_scheduler_predictor.py``.

Note: This module does **not** integrate with vLLM directly.  It is
intended to illustrate how one might build an admission controller
around vLLM's scheduler by modelling step times.  Production
deployments would need to wire the call‑sites appropriately.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class RequestFeatures:
    """Metadata about a single waiting request.

    Attributes
    ----------
    req_id: str
        An opaque identifier for the request.  Not used by the model
        itself but useful for higher‑level bookkeeping.
    prompt_len: int
        The length of the prompt (context) in tokens.
    context_len: int
        Alias for ``prompt_len`` retained for backwards compatibility.
    num_new_tokens: int
        Number of new tokens to compute for this request (prefill or
        chunked prefill).  For decode steps this is typically zero.
    max_tokens: int
        Optional upper bound on the total tokens to produce for this
        request.  Not used by the base model but stored for
        completeness.
    priority: int
        Optional priority level.  Unused here but could be fed into
        more advanced controllers.
    is_prefill: bool
        Whether this request represents a prefill operation.  When
        ``True`` the ``num_new_tokens`` represent prefill tokens.
    arrival_ts: float
        Timestamp when the request entered the system (seconds since
        epoch).  Unused by the model but stored for completeness.
    """

    req_id: str
    prompt_len: int
    context_len: int
    num_new_tokens: int
    max_tokens: int = 0
    priority: int = 0
    is_prefill: bool = False
    arrival_ts: float = 0.0


@dataclass
class StepFeatures:
    """Features summarising a single scheduler step.

    These features describe the aggregate properties of the batch of
    requests currently running.  They form the independent variables
    for the regression model.

    Attributes
    ----------
    ts: float
        Timestamp when the step began (seconds since epoch).
    num_running_reqs: int
        Number of requests currently in the running batch.
    decode_tokens_total: int
        Total number of tokens to be decoded in this step.  For pure
        decode steps this is the size of the decode batch.  For
        prefill‑only steps this will be zero.
    prefill_tokens_total: int
        Total number of prefill or chunked prefill tokens being
        computed in this step.  For decode‑only steps this is zero.
    num_prefill_reqs: int
        Number of requests in this step that are performing prefill.
    max_context_len: int
        Maximum context length among the running decode requests.
    decode_avg_context_len: float
        Average context length for decode requests.
    prefill_avg_seq_len: float
        Average sequence length for prefill requests (i.e., average number
        of prefill tokens per prefill request). When 0 or not set, computed
        as prefill_tokens_total / num_prefill_reqs.
    world_size: int
        Effective world size (e.g. tensor parallel size * pipeline
        parallel size).  This can be used to normalise features across
        different distributed configurations.  It defaults to 1.  The
        caller may supply other values but the controller will never
        derive the value from node counts directly.
    """

    ts: float
    num_running_reqs: int = 0
    decode_tokens_total: int = 0
    prefill_tokens_total: int = 0
    num_prefill_reqs: int = 0
    max_context_len: int = 0
    world_size: int = 1
    # Separate statistics for decode vs prefill to avoid confusion
    decode_avg_context_len: float = 0  # Average context len of decode requests
    prefill_avg_seq_len: float = 0  # Average sequence len of prefill requests

    @classmethod
    def new(cls) -> "StepFeatures":
        return StepFeatures(
            ts=time.time(),
        )

    def clone(self) -> "StepFeatures":
        return StepFeatures(
            ts=self.ts,
            num_running_reqs=self.num_running_reqs,
            decode_tokens_total=self.decode_tokens_total,
            prefill_tokens_total=self.prefill_tokens_total,
            num_prefill_reqs=self.num_prefill_reqs,
            max_context_len=self.max_context_len,
            world_size=self.world_size,
            decode_avg_context_len=self.decode_avg_context_len,
            prefill_avg_seq_len=self.prefill_avg_seq_len,
        )

    def add_decode_request(self, prompt_len: int) -> None:
        self.num_running_reqs += 1
        self.decode_tokens_total += 1  # decode tokens are typically counted separately
        self.max_context_len = max(self.max_context_len, prompt_len)
        self.decode_avg_context_len = (
            self.decode_avg_context_len * (self.num_running_reqs - 1) + prompt_len
        ) / self.num_running_reqs

    def add_prefill_request(self, prompt_len: int, num_new_tokens: int) -> None:
        self.num_running_reqs += 1
        self.prefill_tokens_total += num_new_tokens
        self.num_prefill_reqs += 1
        self.max_context_len = max(self.max_context_len, prompt_len)
        self.prefill_avg_seq_len = (
            self.prefill_avg_seq_len * (self.num_prefill_reqs - 1) + num_new_tokens
        ) / self.num_prefill_reqs

    def has_decode(self) -> bool:
        return self.decode_tokens_total > 0

    # Derived features for better prediction
    @property
    def decode_compute_cost(self) -> float:
        """Approximate decode computation cost: decode_tokens * decode_avg_context_len.

        This captures the O(n*m) complexity of attention where n=decode_tokens
        and m=decode_avg_context_len (context for decode requests).
        """
        return float(self.decode_tokens_total) * self.decode_avg_context_len

    @property
    def prefill_compute_cost(self) -> float:
        """Approximate prefill computation cost.

        For prefill, each token attends to all previous tokens in its sequence,
        giving O(n²) complexity. We approximate total cost as:
        prefill_tokens_total * avg_seq_len_per_prefill_request.

        This is exact when all prefill requests have the same length,
        and a reasonable approximation otherwise.
        """
        avg_seq_len = (
            self.prefill_avg_seq_len
            if self.prefill_avg_seq_len > 0
            else (float(self.prefill_tokens_total) / max(self.num_prefill_reqs, 1))
        )
        return float(self.prefill_tokens_total) * avg_seq_len

    @property
    def kv_cache_size(self) -> float:
        """Approximate total KV cache size: num_running_reqs * decode_avg_context_len."""
        return float(self.num_running_reqs) * self.decode_avg_context_len

    @property
    def prefill_ratio(self) -> float:
        """Ratio of prefill tokens to total tokens."""
        total = self.prefill_tokens_total + self.decode_tokens_total
        return float(self.prefill_tokens_total) / max(total, 1)

    @property
    def avg_decode_tokens_per_req(self) -> float:
        """Average decode tokens per decode request."""
        num_decode_reqs = max(self.num_running_reqs - self.num_prefill_reqs, 1)
        return float(self.decode_tokens_total) / num_decode_reqs

    @property
    def avg_prefill_tokens_per_req(self) -> float:
        """Average prefill tokens per prefill request."""
        return float(self.prefill_tokens_total) / max(self.num_prefill_reqs, 1)


@dataclass
class StepObservation:
    """An observation of a scheduler step and its measured outcomes.

    Attributes
    ----------
    features: StepFeatures
        The input features recorded at the beginning of the step.
    step_time_ms: float
        Total wall‑clock time of the step in milliseconds.
    """

    features: StepFeatures
    step_time_ms: float


class SlidingWindowDataCollector:
    """Collects step observations with an optional sliding window.

    The collector stores up to ``window_size`` observations.  When
    ``decode_only`` is ``True``, only observations where
    ``decode_tokens_total > 0`` are retained.  This allows focusing
    model training on decode steps.  Observations with prefill tokens
    only are dropped.  If insufficient decode observations exist the
    model may still see a small amount of prefill‑only data; callers
    can choose to disable this filter entirely by setting
    ``decode_only=False``.
    """

    def __init__(self, window_size: int = 512, decode_only: bool = False) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.decode_only = decode_only
        self._observations: deque[StepObservation] = deque()

    def update(self, obs: StepObservation) -> None:
        """Add an observation to the collector, respecting the decode filter and
        window size.

        Parameters
        ----------
        obs: StepObservation
            The observation to record.
        """
        # If configured to collect only decode steps, drop observations
        # with zero decode tokens.  Prefill‑only observations still
        # contribute implicitly through their effect on subsequent decode
        # times, but they are not fed to the model directly.
        if self.decode_only and obs.features.decode_tokens_total <= 0:
            return
        self._observations.append(obs)
        # Pop from the left if we exceed the window
        while len(self._observations) > self.window_size:
            self._observations.popleft()

    def __len__(self) -> int:
        return len(self._observations)

    def dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Return feature matrix X and target vector y for regression.

        The features used are a subset of ``StepFeatures`` fields.  To
        reduce variance across large ranges we normalise certain
        dimensions by dividing by constants.  All features are cast to
        floats.

        Returns
        -------
        X: ndarray of shape (n_samples, n_features)
            Feature matrix suitable for a linear model.
        y: ndarray of shape (n_samples,)
            Target vector of step times in milliseconds.
        """
        n = len(self._observations)
        if n == 0:
            return np.empty((0, 0)), np.empty((0,))
        # Define feature extraction: note that world_size is included to
        # allow scaling across distributed setups.  We avoid including
        # node counts directly as per the user requirement.
        X = []
        y = []
        for obs in self._observations:
            f = obs.features
            # Basic normalisation factors to keep magnitudes similar.  These
            # constants can be tuned based on empirical ranges.
            decode_norm = f.decode_tokens_total / 100.0  # ~units of hundreds
            prefill_norm = f.prefill_tokens_total / 100.0  # ~hundreds
            max_ctx_norm = f.max_context_len / 1000.0  # ~thousands
            running_norm = f.num_running_reqs / 10.0  # ~tens
            prefill_reqs_norm = f.num_prefill_reqs / 10.0  # ~tens
            world_norm = f.world_size / 8.0  # world size scaling

            # Separated statistics for decode vs prefill
            decode_ctx_norm = f.decode_avg_context_len / 1000.0  # ~thousands
            prefill_seq_norm = f.prefill_avg_seq_len / 100.0  # ~hundreds

            # Derived features that capture computational complexity
            decode_compute_norm = f.decode_compute_cost / 10000.0  # decode * context
            prefill_compute_norm = (
                f.prefill_compute_cost / 10000.0
            )  # prefill² approximation
            kv_cache_norm = f.kv_cache_size / 1000.0  # total KV cache size
            prefill_ratio = f.prefill_ratio  # already 0-1

            X.append(
                [
                    decode_norm,
                    prefill_norm,
                    max_ctx_norm,
                    running_norm,
                    prefill_reqs_norm,
                    world_norm,
                    decode_ctx_norm,
                    prefill_seq_norm,
                    decode_compute_norm,
                    prefill_compute_norm,
                    kv_cache_norm,
                    prefill_ratio,
                    1.0,
                ]
            )  # bias term
            y.append(obs.step_time_ms)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LinearRegressionModel:
    """Very simple ordinary least squares regression.

    The model recomputes its coefficients from scratch whenever
    ``fit`` is called.  It uses the Moore–Penrose pseudoinverse to
    compute the least‑squares solution.  A small ridge parameter may
    optionally be applied to stabilise the inversion.
    """

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = ridge
        self.coef_: np.ndarray | None = None
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the linear model to the provided data.

        Parameters
        ----------
        X: ndarray
            Feature matrix with shape (n_samples, n_features).
        y: ndarray
            Target vector with shape (n_samples,).
        """
        if X.size == 0:
            self.coef_ = None
            self.n_features_ = 0
            return
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        # Compute (X^T X + ridge I)^{-1} X^T y
        XtX = X.T @ X
        # Add ridge to diagonal to avoid singularity when features are
        # correlated or when n_samples < n_features.
        XtX += self.ridge * np.eye(n_features, dtype=X.dtype)
        XtX_inv = np.linalg.pinv(XtX)
        XtY = X.T @ y
        self.coef_ = XtX_inv @ XtY

    def predict(self, x: np.ndarray) -> float:
        """Predict the target for a single sample.

        Parameters
        ----------
        x: ndarray
            1‑D feature vector with length matching the training matrix
            (including bias term).

        Returns
        -------
        float
            Predicted target value.  If the model has not been
            fitted, returns 0.0.
        """
        if self.coef_ is None or self.coef_.size == 0:
            return 0.0
        return float(np.dot(self.coef_, x))


@dataclass
class AdmissionResult:
    """Result of an admission decision.

    Attributes
    ----------
    allow: bool
        Whether to allow adding the candidate request.
    predicted_step_ms: float
        Predicted total step time if the candidate were admitted.
    predicted_tpot_ms: float
        Predicted per token decode latency (TPOT) in milliseconds per
        token after admission.  This is computed as
        ``predicted_step_ms / max(decode_tokens_after, 1)``.
    reason: str
        Human‑readable explanation of the decision.
    """

    allow: bool
    predicted_step_ms: float
    predicted_tpot_ms: float
    reason: str = ""


class SchedulerPredictor:
    """Admission controller using a linear model of step time.

    This class ties together the data collector and regression model to
    provide a single API for updating the model with observations and
    predicting whether a new request can be admitted without violating
    a step time target.  When insufficient data are available the
    controller defaults to allowing all requests in order to gather
    training samples.
    """

    def __init__(
        self,
        window_size: int = 512,
        decode_only: bool = False,
        step_time_target_ms: float = 50.0,
        ridge: float = 1e-4,
    ) -> None:
        """Construct a new SchedulerPredictor.

        Parameters
        ----------
        window_size: int
            Maximum number of recent observations to keep for model
            fitting.
        decode_only: bool
            Whether to train the model exclusively on observations
            containing decode tokens.  Prefill‑only steps are dropped.
        step_time_target_ms: float
            Desired upper bound on batch step time in milliseconds.
            The predictor will deny admission if it estimates that
            adding the candidate would exceed this value.
        ridge: float
            Ridge regularisation strength used in the linear model to
            improve numerical stability.
        """
        self.collector = SlidingWindowDataCollector(
            window_size=window_size, decode_only=decode_only
        )
        self.model = LinearRegressionModel(ridge=ridge)
        self.step_time_target_ms = step_time_target_ms
        # Fit status flag: we require at least a few samples to form a
        # well‑posed regression.  This threshold can be tuned.  It
        # effectively dictates how many observations are needed before
        # admission decisions become restrictive.
        self._min_samples_for_fit = 10

    def update(self, obs: StepObservation) -> None:
        """Update the predictor with a completed step observation.

        The new observation is stored in the collector.  If enough
        observations are present the linear model is refitted.

        Parameters
        ----------
        obs: StepObservation
            Observed data from the scheduler step.
        """
        # print(f"Updating predictor with observation: {obs}")
        self.collector.update(obs)
        # Only refit if we have enough data to avoid overfitting to
        # extremely small samples.  This helps stabilise early
        # predictions.
        if len(self.collector) >= self._min_samples_for_fit:
            X, y = self.collector.dataset()
            self.model.fit(X, y)

    def _feature_vector(self, f: StepFeatures) -> np.ndarray:
        """Compute the regression feature vector for a given batch.

        This mirrors the logic in ``SlidingWindowDataCollector.dataset`` so
        that the predictor and training are consistent.
        """
        decode_norm = f.decode_tokens_total / 100.0
        prefill_norm = f.prefill_tokens_total / 100.0
        max_ctx_norm = f.max_context_len / 1000.0
        running_norm = f.num_running_reqs / 10.0
        prefill_reqs_norm = f.num_prefill_reqs / 10.0
        world_norm = f.world_size / 8.0

        # Separated statistics for decode vs prefill
        decode_ctx_norm = f.decode_avg_context_len / 1000.0
        prefill_seq_norm = f.prefill_avg_seq_len / 100.0

        # Derived computational complexity features
        decode_compute_norm = f.decode_compute_cost / 10000.0
        prefill_compute_norm = f.prefill_compute_cost / 10000.0
        kv_cache_norm = f.kv_cache_size / 1000.0
        prefill_ratio = f.prefill_ratio

        return np.array(
            [
                decode_norm,
                prefill_norm,
                max_ctx_norm,
                running_norm,
                prefill_reqs_norm,
                world_norm,
                decode_ctx_norm,
                prefill_seq_norm,
                decode_compute_norm,
                prefill_compute_norm,
                kv_cache_norm,
                prefill_ratio,
                1.0,
            ],
            dtype=np.float32,
        )

    def can_admit(
        self,
        current_batch: StepFeatures,
        candidate_prompt_len: int,
        candidate_chunk_tokens: int,
        is_prefill: bool = False,
        world_size: int | None = None,
    ) -> AdmissionResult:
        """Decide whether to admit an additional request.

        A candidate request adds one more running request, contributes
        ``candidate_chunk_tokens`` to the prefill token total, and may
        optionally add ``add_decode_tokens`` to the decode total (e.g.,
        for chunked decode).  The predicted step time after admission
        is compared against the step time target.  If the predicted
        step time would exceed the target, admission is denied.

        If the model does not yet have enough data to fit a
        regression, the method always allows the admission to ensure
        the collector gathers additional observations.

        Parameters
        ----------
        current_batch: StepFeatures
            Features of the current running batch.
        candidate_prompt_len: int
            Context length of the candidate request (used to update
            ``max_context_len`` if larger).
        candidate_chunk_tokens: int
            Number of prefill or chunked prefill tokens the candidate
            would introduce to the next step.
        is_prefill: bool, optional
            Whether the candidate request is a prefill.  This controls
            whether the candidate's tokens are added to the prefill or decode totals.
            Defaults to ``False`` (i.e., decode).
        world_size: int, optional
            Optional override for the world size feature.  When
            provided it will replace ``current_batch.world_size`` in
            the prediction.  If ``None``, the current batch's
            ``world_size`` is used.

        Returns
        -------
        AdmissionResult
            An object encapsulating the decision and prediction.
        """
        # If we do not yet have enough observations to trust the model,
        # always admit to collect more data.  This bootstraps the
        # dataset.
        if len(self.collector) < self._min_samples_for_fit or self.model.coef_ is None:
            return AdmissionResult(
                allow=True,
                predicted_step_ms=0.0,
                predicted_tpot_ms=0.0,
                reason="insufficient data; allowing by default",
            )
        # Build a synthetic step representing the batch after admitting
        # the candidate.  We copy current features and increment
        # relevant fields.  Average and p90 context lengths are left
        # unchanged for simplicity.  Only the max context length and
        # totals matter for this simple model; more complex models
        # could update these statistics more precisely.
        new_world_size = (
            world_size if world_size is not None else current_batch.world_size
        )
        if is_prefill:
            candidate = current_batch.clone()
            candidate.add_prefill_request(candidate_prompt_len, candidate_chunk_tokens)
        else:
            candidate = current_batch.clone()
            candidate.add_decode_request(candidate_prompt_len)
        candidate.world_size = new_world_size
        x = self._feature_vector(candidate)
        predicted_step_ms = self.model.predict(x)
        decode_total_after = max(candidate.decode_tokens_total, 1)
        predicted_tpot_ms = predicted_step_ms / float(decode_total_after)
        # print(f'can_admit: {locals()=}')
        if predicted_step_ms > self.step_time_target_ms:
            return AdmissionResult(
                allow=False,
                predicted_step_ms=predicted_step_ms,
                predicted_tpot_ms=predicted_tpot_ms,
                reason=(
                    f"predicted step time {predicted_step_ms:.3f} ms exceeds "
                    f"target {self.step_time_target_ms:.3f} ms"
                ),
            )
        else:
            return AdmissionResult(
                allow=True,
                predicted_step_ms=predicted_step_ms,
                predicted_tpot_ms=predicted_tpot_ms,
                reason=(
                    f"predicted step time {predicted_step_ms:.3f} ms within target {self.step_time_target_ms:.3f} ms"
                ),
            )

    def max_prefill_tokens_allowed(
        self,
        current_batch: StepFeatures,
        avg_prefill_seq_len: float | None = None,
        max_token_budget: int = 10000,
        world_size: int | None = None,
    ) -> int:
        """Predict the maximum number of prefill tokens that can be added to the current batch.

        This method computes how many prefill tokens can be added while keeping
        the predicted step time below the target. It assumes prefill tokens will
        be added with a given average sequence length.

        Parameters
        ----------
        current_batch: StepFeatures
            Features of the current running batch.
        avg_prefill_seq_len: float, optional
            Average sequence length for new prefill tokens. If None, uses the
            current batch's prefill_avg_seq_len or defaults to 100.
        world_size: int, optional
            Optional override for the world size feature.

        Returns
        -------
        int
            Maximum number of prefill tokens that can be added. Returns 0 if
            the model is not ready or if current batch already exceeds target.
        """
        # If we do not yet have enough observations to trust the model, return a conservative estimate
        if len(self.collector) < self._min_samples_for_fit or self.model.coef_ is None:
            # Without a model, return a conservative default (e.g., allow some prefill)
            return max_token_budget  # Conservative default

        # Predict current step time
        x_current = self._feature_vector(current_batch)
        current_step_ms = self.model.predict(x_current)

        # If already over budget, cannot add more
        if current_step_ms >= self.step_time_target_ms:
            return 0

        # Available time budget
        time_budget_ms = self.step_time_target_ms - current_step_ms

        # Determine the average sequence length for new prefill tokens
        if avg_prefill_seq_len is None:
            avg_prefill_seq_len = (
                current_batch.prefill_avg_seq_len
                if current_batch.prefill_avg_seq_len > 0
                else 100.0
            )
        if not avg_prefill_seq_len:
            avg_prefill_seq_len = 1

        # Binary search to find maximum prefill tokens
        # Start with a reasonable upper bound
        left, right = 0, max_token_budget  # Use the provided max_token_budget
        max_tokens = 0

        while left <= right:
            mid = (left + right) // 2

            # Create a candidate batch with 'mid' additional prefill tokens
            candidate = current_batch.clone()
            if mid > 0:
                # Add prefill tokens (assuming they come as one or more requests)
                # For simplicity, we add them as if they're part of the existing prefill workload
                candidate.prefill_tokens_total += mid
                # Update prefill_avg_seq_len based on the new average
                if candidate.num_prefill_reqs > 0:
                    total_prefill_tokens = current_batch.prefill_tokens_total + mid
                    # Approximate: assume new tokens come in chunks of avg_prefill_seq_len
                    num_new_reqs = max(1, int(mid / avg_prefill_seq_len))
                    candidate.num_prefill_reqs += num_new_reqs
                    candidate.num_running_reqs += num_new_reqs
                    candidate.prefill_avg_seq_len = (
                        total_prefill_tokens / candidate.num_prefill_reqs
                    )
                else:
                    # First prefill request
                    candidate.num_prefill_reqs = 1
                    candidate.num_running_reqs += 1
                    candidate.prefill_avg_seq_len = float(mid)

            if world_size is not None:
                candidate.world_size = world_size

            # Predict step time with this many additional tokens
            x_candidate = self._feature_vector(candidate)
            predicted_step_ms = self.model.predict(x_candidate)

            if predicted_step_ms <= self.step_time_target_ms:
                # Can fit this many tokens, try more
                max_tokens = mid
                left = mid + 1
            else:
                # Too many tokens, try fewer
                right = mid - 1
        print(
            f"max_prefill_tokens_allowed: {current_batch=}, {avg_prefill_seq_len=}, {max_tokens=}"
        )
        return max_tokens
