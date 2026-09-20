# OTTO Decision Lab

从会话行为预测到可解释的推荐决策：比赛复现、多目标实验与服务原型。

项目的核心问题：**在同一份可用信息和相同计算预算下，应该把哪张列表给用户？我们凭什么选择它，又有哪些效果无法由离线数据证明？**

当前状态（2026-09-20）：已安装 Kaggle CLI 2.2.4，下载官方公开数据，完成 1% 稳定会话抽样和两组时间隔离基线，并与固定版本官方评估器对拍一致。[首轮真实实验报告](reports/M0_FIRST_BASELINE.md)。已完成首次赛后提交：Public **0.50587**、Private **0.50768**，见[提交记录](reports/LEADERBOARD_STATUS.md)。尚未训练精排模型；全量算力与预算待确定。

## 开始

```sh
cd /Users/ethan/workspace/projects/kaggle/otto-decision-lab
python3 -m unittest discover -s tests -v
python3 -m otto.evaluate --labels data/validation_labels.jsonl --predictions artifacts/predictions.csv
```

评估器只依赖 Python 3.9+ 标准库。标签每行格式为 `{"session":42,"labels":{"clicks":1,"carts":[2,3],"orders":[3]}}`；clicks 也接受最多一个元素的列表，缺失行为可省略或用空列表。预测格式为 `session_type,labels`，每个标签会话必须提供 clicks、carts、orders 三行；输出取前 20 个位置，再按商品去重，不允许重复会话行或未知会话。大型提交目前载入内存，后续应改成分区评估。

## 阅读顺序

1. [项目方案与验收](docs/PROJECT.md)：完整范围、实验顺序、资源分级、最终交付。
2. [评估与证据协议](docs/EVALUATION.md)：指标、时间隔离、证据边界。
3. [研究日志](docs/RESEARCH_LOG.md)：自己的判断、反证与决策记录。
4. [资料](docs/SOURCES.md)：官方规则和研究对照。
5. [Kaggle CLI 使用](docs/KAGGLE_CLI.md)：数据下载、登录和首轮实验命令。

## 完成标准

真实数据可复現；强基线与逐项消融；统一列表的可行解与 Pareto 比较；服务与策略回退验证；完整实验记录与上线决策备忘录。演示服务不等于已验证的生产系统，离线代理指标不等于实际业务收益。

不预设金牌分数或收益数字。每个阶段通过验收再扩大规模；“极限”指把效果、成本和证据做透，而不是无限堆模型。
