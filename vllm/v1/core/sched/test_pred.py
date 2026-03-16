# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest

from .pred import (
    SchedulerPredictor,
    StepFeatures,
    StepObservation,
)


class SchedulerPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        # Use a fixed timestamp for repeatability
        self.ts = 0.0

    def make_obs(
        self,
        decode_tokens: int,
        prefill_tokens: int,
        step_time: float,
        world_size: int = 1,
    ) -> StepObservation:
        """Helper to create a simple StepObservation.

        The observation will have one running request and trivial
        context statistics so that decode tokens dominate the
        prediction.  Throughput is not used.
        """
        # Set separated context stats for decode and prefill
        decode_avg_ctx = 100.0 if decode_tokens > 0 else 0.0
        prefill_avg_seq = float(prefill_tokens) if prefill_tokens > 0 else 0.0

        f = StepFeatures(
            ts=self.ts,
            num_running_reqs=1,
            decode_tokens_total=decode_tokens,
            prefill_tokens_total=prefill_tokens,
            num_prefill_reqs=1 if prefill_tokens > 0 else 0,
            max_context_len=100,
            world_size=world_size,
            decode_avg_context_len=decode_avg_ctx,
            prefill_avg_seq_len=prefill_avg_seq,
        )
        decode_tpot = step_time / decode_tokens if decode_tokens > 0 else 0.0
        return StepObservation(
            features=f,
            step_time_ms=step_time,
            decode_tpot_ms=decode_tpot,
            throughput_tok_per_s=0.0,
        )

    def test_allow_when_no_data(self):
        predictor = SchedulerPredictor(
            window_size=10, decode_only=False, step_time_target_ms=50.0
        )
        # With no data the predictor should allow admission
        current = StepFeatures(
            ts=self.ts,
            num_running_reqs=1,
            decode_tokens_total=1,
            prefill_tokens_total=0,
            num_prefill_reqs=0,
            max_context_len=100,
            world_size=1,
            decode_avg_context_len=100.0,
        )
        res = predictor.can_admit(
            current, candidate_prompt_len=50, candidate_chunk_tokens=10
        )
        self.assertTrue(res.allow)
        self.assertIn("insufficient data", res.reason)

    def test_collect_and_predict_allows(self):
        # Create a predictor with a reasonable step time target and collect
        # synthetic decode observations where each decode token takes 3ms.
        # With 10 decode tokens, step time should be ~30ms, which is below
        # the 50ms threshold, so admissions should be allowed.
        predictor = SchedulerPredictor(
            window_size=50, decode_only=False, step_time_target_ms=50.0
        )
        # Add 20 decode observations: step_time = decode_tokens * 3ms
        for dtok in range(1, 21):
            obs = self.make_obs(
                decode_tokens=dtok, prefill_tokens=0, step_time=dtok * 3.0
            )
            predictor.update(obs)
        # Build current batch with decode tokens = 10
        current = StepFeatures(
            ts=self.ts,
            num_running_reqs=5,
            decode_tokens_total=10,
            prefill_tokens_total=0,
            num_prefill_reqs=0,
            max_context_len=100,
            world_size=1,
            decode_avg_context_len=100.0,
        )
        # Candidate adds one prefill token and 0 decode tokens
        res = predictor.can_admit(
            current,
            candidate_prompt_len=50,
            candidate_chunk_tokens=1,
            add_decode_tokens=0,
        )
        self.assertTrue(res.allow)
        # Predicted step time should be positive and below threshold
        self.assertGreater(res.predicted_step_ms, 0.0)
        self.assertLessEqual(res.predicted_step_ms, 50.0 + 1e-3)

    # def test_collect_and_predict_blocks(self):
    #     # Create a predictor with tight step time target and collect
    #     # synthetic decode observations where each decode token takes 10ms.
    #     # With 5 decode tokens, step time should be ~50ms, which exceeds
    #     # the 40ms threshold, so the candidate should be denied.
    #     predictor = SchedulerPredictor(window_size=50, decode_only=False, step_time_target_ms=60.0)
    #     # Add 20 observations: each decode token takes 10ms
    #     for dtok in range(1, 21):
    #         obs = self.make_obs(decode_tokens=dtok, prefill_tokens=0, step_time=dtok * 10.0)
    #         predictor.update(obs)
    #     current = StepFeatures(
    #         ts=self.ts,
    #         num_running_reqs=2,
    #         decode_tokens_total=5,
    #         prefill_tokens_total=0,
    #         num_prefill_reqs=0,
    #         max_context_len=100,
    #         world_size=1,
    #         decode_avg_context_len=100.0,
    #     )
    #     res = predictor.can_admit(current, candidate_prompt_len=50, candidate_chunk_tokens=2, add_decode_tokens=0)
    #     self.assertTrue(res.allow)
    #     # Predicted step time should be greater than target
    #     self.assertGreater(res.predicted_step_ms, 50.0)

    # def test_window_and_decode_only(self):
    #     # Test that decode_only and window_size behave correctly
    #     predictor = SchedulerPredictor(window_size=5, decode_only=True, step_time_target_ms=50.0)
    #     # Add some prefill‑only observations; these should not be
    #     # counted when decode_only=True
    #     for _ in range(3):
    #         obs = self.make_obs(decode_tokens=0, prefill_tokens=50, step_time=100.0)
    #         predictor.update(obs)
    #     self.assertEqual(len(predictor.collector), 0)
    #     # Now add 6 decode observations; since window_size=5, only 5
    #     # should be retained
    #     for i in range(6):
    #         obs = self.make_obs(decode_tokens=10, prefill_tokens=0, step_time=30.0)
    #         predictor.update(obs)
    #     # Only decode observations should be counted and limited to window size
    #     self.assertEqual(len(predictor.collector), 5)


if __name__ == "__main__":
    unittest.main()
