# GLM H200 专用执行器封装：v53 完整结果

已实现模型自有 SpecializedDecode / NVIDIA wrapper / MRV2 ModelState / 配置拒绝 / 全 KDA metadata 资格判断 / cache 绑定 / capture 前 scratch。仍为多 kernel 执行，不是全层单 persistent launch。原有通用 hooks 作为冻结基础保留。

测试源码 c3b22ba3a25f3849b3c4e80cbe81ff5c4caecb4b，基线检查点 e65eac9980；full45 FP8 TP8 c1 DFlashK3 Q4BV8，WO-FP8、私有 BF16 draft、共享 WO head、BF16-compatible fused AR 均一致。单次模型加载 ABBA，每阶段3个排除4096 warmup +9正式4096 ignoreEOS代码请求。共36正式+12warmup；每种路径同64道独立GSM8K，不按128道独立题计算置信度。

| 路径 | 18请求合并 tok/s | 最高整请求 | 最低 | AL | 估计cycle ms | >400 / >450 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 继承 | 461.48 | 525.61 | 371.04 | 2.8067 | 6.0804 | 16/18 / 14/18 |
| 专用 | 451.86 | 532.03 | 394.36 | 2.7468 | 6.0773 | 16/18 / 9/18 |

专用峰值532.03来自 specialized_b，同批九请求均值446.18。旧峰值525.61来自legacy_a，同批468.77。阶段顺序及均值468.77→446.18→457.69→454.41。按token总量/decode总时间合并，不平均请求速率。估计cycle=decode秒/投机验证步数，并非CUPTI纯GPU时间。

吞吐−2.08485%，AL−2.13%，估计cycle−0.05093%。观测差异与接受率变化相符，不能直接归为kernel因果回退；但预设吞吐≤1%回退门已失败，因此不推广、不替换auto。平均500、全部请求稳定400/450均未达到。不把532峰值宣传为稳定吞吐。

GSM8K两路均63/64，无截断；同64题完整token序列64/64一致，输入hash一致。小样本不证明普遍精度等价；长代码同路径同prompt三次生成每组均3个不同完整输出，跨阶段与首legacy9请求比较0/27完整输出一致，生成轨迹已有波动。不能因同数学分数就宣称长代码逐token恒等。

8rank审计全部通过：新ModelState、34KDA prepared、generation2捕获；A/D同一legacy图，B/C同一specialized图，两图身份独立，weights/KV/drafter/prepared/arrivals身份稳定，源码hash相同，三prompt token IDs一致。正式trace invalid0。八卡1Hz遥测无采样错误/显著显存增长告警；这是有限快照及聚合检测，不证明不存在所有外部干扰。测试前后源码hash全部一致。实验结束八卡0MiB/0%。

v51加载后发现多模态外壳绕过CausalLM.forward，0正式；v52发现pre-KV profiling先检查cache，0正式。均有RED→GREEN回归、独立目录和压缩原始失败记录。未删除负结果。

上游并非无人做：AMD GLM5.3 Mono draft#61662针对MI355X/MXFP4/非投机；Kimi#61867已合并；NVIDIA有KDA/mHC单项优化。检索暂未发现同范围H200/FP8/TP8/DFlash完整专用执行路径，不断言未公开工作。没有开PR或改共享分支。

原始数据runtime_abba.json、all4 telemetry、source_manifest/postrun hashes、package_versions、frozen overlay、完整/封装patch与分析脚本均归档。Site同项目公开更新。
