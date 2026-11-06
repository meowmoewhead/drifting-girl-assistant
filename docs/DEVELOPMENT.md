# 源码运行与构建

## 验证环境

Windows x64、Python 3.14 x64，运行及构建依赖按仓库中的版本文件安装。其他 Python 版本尚未由本次发布验证。

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
```

源码检出包含数字识别模型，模型文件及来源/校验值位于 `models/`。如模型缺失，可运行 `dev_tools/fetch_digit_model.py` 下载固定模型并校验；正常识别在本地执行。MuMu SDK DLL 使用本机模拟器安装目录中的版本，不随项目复制。

## 源码启动

首次启动会从 `config/default.json` 创建缺失的 `config.json`；已有配置保持不变。可在桌面面板选择模拟器。然后运行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 desktop_entry.py
```

也可双击 `启动.cmd`。打开面板不会自动开始任务。

## 离线检查

```powershell
.\.venv\Scripts\python.exe -B -X utf8 deploy/verify_release.py
.\.venv\Scripts\python.exe -B -X utf8 -m unittest tests.test_public_distribution tests.test_package_release tests.test_version tests.test_task_registry
.\.venv\Scripts\python.exe -B -X utf8 app.py framework-check
```

这些检查核对默认关闭、完整规则/模板、模型、打包隔离、版本及 Maa 任务执行，不连接模拟器。发布前也会拒绝测试素材清单中的绝对路径，素材来源只保留文件名、相对路径或公开链接。没有构建产物时，直接读取真实 EXE 的一项测试会明确跳过，合成测试 EXE 的打包测试仍运行。

完整回归可使用：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 dev_tools/test_gate.py diagnostics/full-regression
```

每个测试模块在独立进程中运行，最多两个并发，输出有界进度及内存统计。部分历史实机取证图没有公开，相应测试明确跳过；不要把跳过项当作已验证。核心单元与 `tests/fixtures` 中分发的回放样例均保留。旧收集目录中的12张必要测试图已迁移到 `tests/fixtures/legacy-captures`。

## 构建便携版

```powershell
.\.venv\Scripts\python.exe -B -X utf8 deploy/build_release.py
```

或运行 `powershell -ExecutionPolicy Bypass -File deploy/build.ps1`。需要额外运行全部回归时加 `--full-tests`，PowerShell 入口对应 `-FullTests`。

默认构建会执行发布契约测试、Python解析/模型/模板检查，使用新的构建目录调用 PyInstaller，再运行产物的版本与 Maa 检查，最后校验 ZIP。默认检查不声称替代全部游戏流程的实机验证。

输出位于 `release/`：

- `mumu-panel.exe`
- `mumu-clicker-0.2.0-win64.zip`
- `build-report.json`

便携包内 `config.json` 始终来自 `config/default.json`，本机的配置、状态、截图和日志不会成为发布输入。所有规则引用的模板变体都必须存在，不会为了缩减体积删除配置仍引用的活动图片。

当前结构：`drift/core` 基础设施、`drift/device` 设备通道、`drift/ui` 页面感知、`drift/perception` 数字识别、`drift/tasks` 任务、`drift/web` 面板与CLI；根目录兼容模块用于既有导入路径。`alas_core` 保存固定上游基础组件的改编版。

完整目录职责见 [项目目录](STRUCTURE.md)。
