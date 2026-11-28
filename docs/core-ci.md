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

## Wheel 构建与隔离安装

CI 在源码验收后增加 wheel 构建及安装验证。构建使用固定 setuptools 81.0.0 和 wheel 0.46.3，禁用构建隔离与依赖解析；构建产物进入 outputs/ci/wheels。该策略显式使用 CI 已安装的构建工具，不验证从 sdist 构建或完整依赖锁定。

验证脚本 scripts/verify_wheel.py 接受一个可信本地 wheel 和不存在的输出目录。它创建全新临时虚拟环境，清除 PYTHONPATH/PYTHONHOME，禁用用户包目录，从仓库外的工作目录运行。安装使用 --no-index --no-deps，不下载机器人/模型依赖。检查导入位置确实处于该虚拟环境、tabletop.xml 包资源存在、三个 console entry point 的 --help 可用，以及默认 mock 任务完成四步。

```sh
.venv/bin/python -m pip wheel --no-deps --no-build-isolation --wheel-dir outputs/local-wheels .
.venv/bin/python scripts/verify_wheel.py \
  --wheel outputs/local-wheels/embodied_agent-0.1.0-py3-none-any.whl \
  --output outputs/local-wheel-check
```

wheel-verification.json 保存 wheel 摘要、阶段退出码和输出，失败也尽可能留下报告；临时虚拟环境退出后清理。已有结果目录拒绝覆盖，子命令设 120 秒超时。失败会阻止 CI 成功，wheel、构建日志和验证报告沿用已有产物上传步骤。此脚本会执行安装包代码，仅用于本仓库构建的可信包，不是恶意包沙箱。

这仍是 CI 构建验证，不发布到 PyPI、不部署设备。XML 资源检查不等于 MuJoCo 实际运行，mock 成功不代表实机成功。新增三项测试覆盖缺失 wheel、拒绝覆盖和失败报告；277 项核心测试本地通过。云端构建与上传需 push 后确认。

## 安装后的持久化执行验收

Wheel 验证进一步增加五个真实子进程阶段：创建 quarantine 预算、带 operation-id 执行四步 mock 任务、同 ID 重试返回历史结果、同 ID 不同指令退出 3 并报告 operation_conflict、独立进程读取预算。它们均使用新虚拟环境中的安装包，清除源码路径并从仓库外运行。

首次执行要求 replayed:false，重复请求要求 replayed:true；重复、冲突及检查之后，保存的预算和回执 JSON 必须与首次执行后逐字节一致。验证器不会仅凭退出 0 判定功能正常。报告记录每步实际与预期退出码，并保留 managed-state 下的预算、回执和对应 SHA-256，随 CI artifact 上传。临时 Python 环境仍在结束后清理，模拟状态作为验收证据保留。失败时尽可能保存已有阶段和产物摘要。

这验证安装后的持久化接口和重复请求处理，不是机械臂物理 exactly-once，也不覆盖断电耐久性或并发竞争。超时、磁盘故障等仍可能导致报告不完整。新增两项合同测试（包含写入变化、错误重放标记及错误冲突原因三个故障变体），279 项核心测试本地通过；重新构建的 wheel 在新虚拟环境完成全部 11 个安装验收阶段。云端执行仍待 push。
