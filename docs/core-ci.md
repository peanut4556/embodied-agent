# 核心 GitHub Actions CI

本轮新增 `.github/workflows/core-ci.yml`，将已有本地检查接入 GitHub Actions。代码配置已完成并在本地验证；推送前没有云端运行记录，不能把本地成功表述为 GitHub CI 已通过。

## 触发和工作内容

向 main、codex/** 分支 push、提交或更新 PR、手动 workflow_dispatch 均可触发。运行环境为 ubuntu-24.04、Python 3.12，单作业超时 15 分钟，同一 ref 的新运行取消旧运行。PR 使用 pull_request，不使用 pull_request_target。令牌权限为 contents:read，checkout 不保留凭据，不配置部署密钥。

执行顺序：

1. 安装 requirements/ci.txt 固定的 Ruff 0.16.6，缓存 pip 下载。
2. 对 src/embodied_agent、scripts、tests 执行 Ruff 检查和格式检查。
3. 根据 config/ci-core-tests.txt 的 40 个明确模块执行核心回归，目前 266 项。
4. 在 outputs/ci/lifecycle 运行 11 步真实 CLI 维护验收。
5. 对本次产物执行源码摘要和阶段结果复核。
6. 无论前面成功或失败，都尝试上传 outputs/ci，保留 14 天；如果准备目录前失败则可能无产物。

Bash 显式作为运行 shell，Actions 的 Bash 执行使用失败及管道失败传播，tee 保存日志不会吞掉测试失败。没有 continue-on-error。前面的阶段失败时后续业务步骤停止，上传步骤仍尝试执行。产物名含 run_id 和 run_attempt，便于区分重新运行。

Actions 版本采用当前官方文档的 v7 主版本引用：[checkout](https://github.com/actions/checkout)、[setup-python](https://github.com/actions/setup-python)、[upload-artifact](https://github.com/actions/upload-artifact)。主版本标签并非不可变 SHA；需要完全固定第三方 Action 时可另行锁定提交。Python 补丁版本和托管运行器也未完全冻结。

## 本地复现

```sh
python3.12 -m venv .venv-ci
.venv-ci/bin/python -m pip install -r requirements/ci.txt
make ci-lint CI_PYTHON=.venv-ci/bin/python
make ci-test CI_PYTHON=.venv-ci/bin/python

mkdir -p outputs/ci-local
PYTHONPATH=src:. .venv-ci/bin/python scripts/validate_recalibration_lifecycle.py \
  --output outputs/ci-local/lifecycle
PYTHONPATH=src:. .venv-ci/bin/python -m embodied_agent.recalibration_evidence \
  --directory outputs/ci-local/lifecycle --source-root . --check-stage-results
```

验收子目录必须不存在，重复运行时使用新的目录；不要覆盖旧证据。CI_PYTHON 默认 .venv/bin/python，云端传入 python。清单模块全部依赖标准库与项目源码，不需要 pip 安装项目或重型可选依赖。新增核心测试需要显式加入清单，未加入的测试不会自动被本工作流执行。

本地验证使用全新 Python 3.12.14 虚拟环境，核心测试 266 项通过，11 步验收及完整复核通过。Ruff 使用原工作环境的同版本 0.16.6，全检查范围 535 个 Python 文件格式通过，并修正一处既有脚本换行。工作流通过 YAML 解析和触发器/权限/上传条件检查；尚未使用 actionlint 或 GitHub 托管运行器实际执行。当前本机 Docker 服务未运行，未验证容器流水线。

## 查看结果与边界

push 后进入仓库 Actions → Core CI，查看 Core checks (Python 3.12) 作业。失败时先定位失败步骤，再下载对应 artifact 中的 lint.log、format.log、tests.log、lifecycle.log、verification.json 及 lifecycle 下的原始状态和 report.json。前序失败意味着部分后续文件不存在，这是正常现象；缺失不能当作通过。

这轮是核心 CI，不包括 LeRobot/RAI 模型下载、ROS 2 镜像构建、MuJoCo 仿真、真实机器人测试、自动发布或部署（CD）。仓库已有的这些能力仍通过各自手动命令使用。本工作流也不自动设置分支保护；如需合并前强制通过，应在远端规则中将该检查设为必需。尚未修改远端设置，是否允许 Actions 运行仍取决于仓库权限与策略。

面试表述应区分状态：当前可以说“实现了 CI 工作流并完成本地等价步骤验证”；首次远端成功后才能说“GitHub Actions 已实际跑通”。报告校验仍是内容一致性检查，不认证来源或实机结果。

## 结构化测试报告

CI 核心回归现在通过 scripts/run_core_ci.py 执行清单，当前 41 个模块、269 项测试。保留原 make ci-test，并新增 `make ci-report CI_REPORT_DIR=outputs/ci-local/new-results`。输出目录必须不存在，避免覆盖旧报告。

报告包含 tests.json 和 summary.md：记录通过状态、测试数量、耗时、失败和错误的用例 ID 与堆栈、跳过原因、预期失败和意外成功。子测试失败也保留其标识。GitHub Step Summary 展示计数与结果，详细堆栈在 artifact 中；测试失败仍返回非零，上传步骤不改变失败结论。

清单必须非空、模块名合法且无重复；导入失败或没有用例时直接失败，不输出伪成功报告。测试执行完成后的普通失败会保存报告；进程被杀、超时或初始化失败可能没有报告，此时页面明确提示查看前序步骤，不把缺失当作成功。报告是 unittest 汇总而非 JUnit XML，不记录每个成功用例的独立耗时。

新增三项报告行为测试，本地核心回归 269 项通过；云端摘要和上传行为仍需 push 后实际确认。

## 初始化与运行器故障报告

ci-report 现在先独占创建新输出目录，再加载清单。清单缺失、非法、导入异常等普通初始化失败会保存 runner-error.json，记录 phase、cause_type 和 traceback，并生成 RUNNER ERROR 摘要；正常的测试断言失败仍记录在 tests.json。直接调用脚本时：通过退出 0，测试失败退出 1，已捕获的运行器故障退出 3；通过 make 调用时 make 可能将非零配方状态映射为其自身退出码，但 CI 仍失败。

清单最多 65536 个字符、256 个模块，逐模块检查用例数量；一个空模块不能被其他正常模块掩盖。导入失败会保留模块名和 loader 错误。已有输出目录不覆盖，目录创建失败发生在报告路径可用之前，此时仍需查看日志。

报告生成或执行中的普通异常标为 execute_or_report；已有 tests.json 不一定代表整个运行成功，应优先检查 runner-error.json 与进程退出码。磁盘不可写时故障报告本身也可能无法保存；强制退出、KeyboardInterrupt 和超时不承诺有报告。堆栈可能含本地路径与异常文本，产物沿用仓库访问权限。

新增五项测试验证清单缺失、单模块无用例、运行器异常及不覆盖、真实 CLI 导入失败和大小限制。当前 274 项核心测试通过，云端验证仍待 push。
