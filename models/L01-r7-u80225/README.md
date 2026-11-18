# L01 r7 第 80225 次更新：r7 第一次原版确认

来源：训练 r7，第 80225 次更新。

## 评估

- 原版从出生点起评估（温度 0.3）：通关 19 / 20，无死亡 4；用时 最快 89.318 s、中位 116.719 s、最慢 150.924 s
- 确定性按键：LevelFinish，94.765 s，死亡 0
- 确定性按键：LevelFinish，94.765 s，死亡 0

- `L01_89_318_r7-u80225.tas`：89.318 s，真实 TASSupport 重放 5/5 一致（PASS）
- `L01_89_386_r7-u80225.tas`：89.386 s，真实 TASSupport 重放 5/5 一致（PASS）

## 文件

- `L01-r7-u80225.pt`：PyTorch 检查点（网络、观测归一化、训练参数、环境配置；无优化器状态，路径为仓库相对路径），可直接用于 `scripts/eval_m2.py --ckpt`。
- `plain/<sha>.npz` + `.json`：不用 pickle 的纯数组导出（`ballrain/pool/weights.py`），文件名是策略的 sha256。
