# 实验报告：sklearn digits 手写数字分类

## 选择

在验证集 **macro-F1** 上比较三个候选：

| 模型 | 验证 accuracy | 验证 macro-F1 |
|------|---------------|---------------|
| DummyClassifier（most_frequent） | 0.1003 | 0.0182 |
| StandardScaler + LogisticRegression | 0.9666 | 0.9665 |
| RandomForestClassifier（n_estimators=200） | **0.9694** | **0.9693** |

**选定：`random_forest`**（验证 macro-F1 最高）。随后在 **train+val** 上重新拟合同一配置，**仅一次**在测试集上报告。

## 数据与划分

- 数据：`sklearn.datasets.load_digits`，共 1797 样本、64 维像素特征、标签 0–9。
- `train_test_split` 两次、`stratify`、`random_state=42`：
  - train 1078（≈59.99%）
  - val 359（≈19.98%）
  - test 360（≈20.03%）
- 候选比较阶段：预处理器/模型只在 **train** 上 fit；测试集不参与任何选型。

## 最终测试结果（真实运行）

| 指标 | 数值 |
|------|------|
| accuracy | 0.9666666666666667 |
| macro-F1 | 0.9665384091991565 |
| 最终模型训练耗时（train+val） | 0.31805169800099975 s |

目标准确率 95%：**已达到**（约 96.67%）。

预测明细见 `results/test_predictions.csv`（360 行测试样本）。完整日志见 `logs/run.log`。

## 问题与说明

1. 本环境 `scikit-learn==1.9.1` 中 `LogisticRegression` 已移除 `multi_class` 参数；实现中不传该参数，避免 TypeError。
2. Dummy 基线按「最频繁类」策略，验证准确率约 10%，符合十类均衡数据预期。
3. LogisticRegression 与 RandomForest 验证指标非常接近；按既定准则（val macro-F1）仍选 RandomForest。
4. `.gitignore` 排除 `.venv/`、缓存与凭证类文件；保留 `logs/run.log`。

## 局限

- 未做系统超参搜索（网格/贝叶斯等）；RandomForest 固定 `n_estimators=200`，LogReg 固定 `max_iter=5000`。
- digits 为小规模、低分辨率玩具数据，结论不宜外推到真实拍照/扫描数字场景。
- 未比较 SVM、梯度提升或 CNN 等更强模型；未做校准或置信度分析。
- 墙钟时间依赖机器负载与 `n_jobs=-1`；数值以本机一次运行为准，复现应得到相同预测与指标（给定相同依赖版本与种子）。
