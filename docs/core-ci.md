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
