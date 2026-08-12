# 语音克隆 SIM 数据汇总

数据来源：当前运行后端中全部已完成且同时具有原始克隆 SIM、防护后克隆 SIM 的记录，共 38 条，覆盖 4 个当前支持的克隆模型。已移除的 XTTS-v1.1 不纳入统计。

SIM 使用 ECAPA-TDNN 声纹嵌入的余弦相似度，理论范围为 `[-1, 1]`：

- 原语音克隆 SIM 越高，表示未防护参考音频的克隆身份越接近原说话人。
- 防护后克隆 SIM 越低，表示保护音频生成的克隆身份越偏离原说话人。
- 绝对下降量：$\Delta \mathrm{SIM}=\mathrm{SIM}_{\mathrm{original}}-\mathrm{SIM}_{\mathrm{protected}}$。
- 相对下降率：$r_{\mathrm{drop}}=\Delta \mathrm{SIM}/\mathrm{SIM}_{\mathrm{original}}\times 100\%$。

## 1. 各模型的历史极值

本表分别统计每个模型的“原语音克隆 SIM 最高值”和“防护后克隆 SIM 最低值”。两个极值允许来自不同实验记录，因此不可将同一行的两个数直接用于计算下降量。

| 模型 | 原语音克隆 SIM 最高值 | 对应记录 | 防护后克隆 SIM 最低值 | 对应记录 |
|---|---:|---|---:|---|
| XTTS-v2 | 0.5443 | `task_6ed27d226465 / clone_b72954e5` | 0.1261 | `task_6ed27d226465 / clone_59d91105` |
| YourTTS | 0.5337 | `task_1baf9c640f33 / clone_23e76a32` | 0.0128 | `task_6ed27d226465 / clone_e7fc8848` |
| CosyVoice2-0.5B | 0.8301 | `task_1baf9c640f33 / clone_c3b6ae7d` | -0.0356 | `task_315001019e8c / clone_94a3eabe` |
| GPT-SoVITS | 0.7506 | `task_b72d4809b785 / clone_4615b999` | 0.0930 | `task_1baf9c640f33 / clone_cff9ce9e` |

> CosyVoice2-0.5B 的防护后 SIM 为负值是合法结果。余弦相似度允许小于 0，表示声纹向量方向已经明显偏离原说话人。

## 2. 各模型同一次实验中 SIM 下降最大的数据

本表每行均来自同一条真实克隆记录，可以直接比较并计算下降量。按绝对下降量 $\Delta \mathrm{SIM}$ 选择每个模型的最大记录。

| 模型 | 原语音克隆 SIM | 防护后克隆 SIM | SIM 绝对下降量 | SIM 相对下降率 | 记录 |
|---|---:|---:|---:|---:|---|
| XTTS-v2 | 0.5443 | 0.1431 | 0.4011 | 73.71% | `task_6ed27d226465 / clone_b72954e5` |
| YourTTS | 0.2983 | 0.0371 | 0.2612 | 87.55% | `task_4209545a2d39 / clone_5641ddbc` |
| CosyVoice2-0.5B | 0.8301 | 0.0502 | 0.7799 | 93.95% | `task_1baf9c640f33 / clone_c3b6ae7d` |
| GPT-SoVITS | 0.7336 | 0.2175 | 0.5161 | 70.35% | `task_b72d4809b785 / clone_fb99f00f` |

## 3. 简要结论

- 原语音克隆 SIM 最高：CosyVoice2-0.5B，最高为 `0.8301`。
- 防护后克隆 SIM 最低：CosyVoice2-0.5B，最低为 `-0.0356`。
- 同一次实验中绝对下降量最大：CosyVoice2-0.5B，由 `0.8301` 降至 `0.0502`，下降 `0.7799`。
- 同一次实验中相对下降率最高：CosyVoice2-0.5B，下降率为 `93.95%`。
