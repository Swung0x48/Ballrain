# L01 r6 第 64400 次更新：r7 之前前段较好的版本（后来回退到这里）

来源：训练 r6，第 64400 次更新。

## 评估

- 原版从出生点起评估（温度 0.3）：通关 12 / 20，无死亡 6；用时 最快 95.393 s、中位 99.560 s、最慢 127.227 s
- 确定性按键：LevelFinish，96.015 s，死亡 0
- 确定性按键：LevelFinish，96.015 s，死亡 0

- `L01_95_393_r6-u64400.tas`：95.393 s，真实 TASSupport 重放 5/5 一致（PASS）
- `L01_95_492_r6-u64400.tas`：95.492 s，真实 TASSupport 重放 5/5 一致（PASS）

## 文件

- `L01-r6-u64400.pt`：PyTorch 检查点（网络、观测归一化、训练参数、环境配置；无优化器状态，路径为仓库相对路径），可直接用于 `scripts/eval_m2.py --ckpt`。
- `plain/<sha>.npz` + `.json`：不用 pickle 的纯数组导出（`ballrain/pool/weights.py`），文件名是策略的 sha256。
