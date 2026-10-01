# research-practice-digits

基于 `sklearn.datasets.load_digits`（手写数字 0–9）的可复现分类练习。固定随机种子 42，分层划分约 60/20/20（训练/验证/测试），在验证集 macro-F1 上比较 Dummy、StandardScaler+LogisticRegression、RandomForest，优胜者用 train+val 重训后仅评估一次测试集。

## 环境

- Python 3.13+（本机实测 3.13）
- 依赖见 `requirements.txt`（已钉死实际安装版本）

## 安装与一键复现

```bash
cd research-practice-digits
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py 2>&1 | tee logs/run.log
```

成功后会写入：

- `results/metrics.json` — 划分规模、各候选验证指标、最终测试指标与训练耗时
- `results/test_predictions.csv` — 测试样本索引、`y_true`、`y_pred`
- 终端输出同步进 `logs/run.log`

## 结果摘要（本仓库一次真实运行）

| 项目 | 数值 |
|------|------|
| 选定模型 | `random_forest`（验证 macro-F1 最高） |
| 测试准确率 | 0.9667 |
| 测试 macro-F1 | 0.9665 |
| 最终训练耗时 | ≈0.32 s |

详情见 `REPORT.md` 与 `results/metrics.json`。
