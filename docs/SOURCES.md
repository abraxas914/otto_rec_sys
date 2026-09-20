# 资料与用途

核验日期：2026-09-20。外部页面和协议可能变化，正式复现时固定 commit 与数据版本。

- [比赛数据说明](https://www.kaggle.com/competitions/otto-recommender-system/data)：字段、未来行为目标和下载体积。
- [官方比赛协议 KAGGLE.md](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md)：评分公式、下一次点击标签、提交格式与允许使用测试前缀。
- [官方数据仓库](https://github.com/otto-de/recsys-dataset)：赛后研究数据、时间切分和匿名化边界。研究协议不应与竞赛协议混淆。
- [MultiTRON 官方实现](https://github.com/otto-de/MultiTRON)：RecSys 2024 偏好条件化与 Pareto 前沿近似研究，作为 M3 的进阶对照；也链接了团队 2025 年离线到线上指标研究。

官方论文与已有代码是基线来源，不代表本项目已经复现其结果；作者的线上结果也不是本项目的线上证据。


2024–2026 序列推荐、生成式推荐与适配判断的原论文/作者仓库清单见 [前沿方法调研](../reports/MODERN_METHODS_REVIEW.md)。本轮已实现其中 PCTM 的 OTTO 适配；其余条目仅完成调研，不标为已复现。
