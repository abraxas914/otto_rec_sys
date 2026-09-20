# Kaggle CLI 工作流

本项目已安装 Kaggle CLI 2.2.4，使用 Python 3.12 独立环境。依赖固定在 requirements.lock.txt。凭据目录 .local/kaggle 被版本控制忽略；不在日志打印 token。

在项目根目录运行：

```sh
sh scripts/kaggle.sh --version
sh scripts/kaggle.sh datasets files otto/recsys-dataset --format json
sh scripts/kaggle.sh datasets download otto/recsys-dataset -p data/raw
.venv/bin/python -m otto.prepare
.venv/bin/python -m otto.baseline
PYTHONPATH=. .venv/bin/python scripts/compare_official.py
```

公开数据文件清单和下载已验证可匿名访问。赛后公开数据保留完整未来行为，和原竞赛截断测试文件不同。本轮只读取训练内容，公开测试集留作后续冻结评估。

需要账号的操作（例如访问要求参赛的文件、运行个人 Notebook 或提交）再登录：

```sh
sh scripts/kaggle.sh auth login
```

登录在浏览器由账号本人完成；不需要向聊天发送密钥。比赛是否仍支持提交、是否需接受规则，以执行时的官方状态为准。

CLI 负责资源获取和远端任务管理，不会自动完成本地训练；后续使用 Kaggle 算力需要准备 Notebook 配置并检查账号配额。

首轮采样：对训练文件所有会话 ID 用 BLAKE2b(seed:session) 做 1% 稳定抽样，完整保留入选会话。验证区间起点为样本最大时间戳加 1 减 7 天；仅起始于区间内的会话参与验证，更早开始的跨界会话只保留界前训练事件。每个验证会话仅在时间严格增长的位置稳定截断。开发样本指标不与正式全量结果比较。
