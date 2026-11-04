param([switch]$FullTests)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw '先按 README 创建 .venv 并安装 requirements-build.txt。' }
$taskArguments = @('-B','-X','utf8',(Join-Path $PSScriptRoot 'build_release.py'))
if ($FullTests) { $taskArguments += '--full-tests' }
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw "构建失败（$LASTEXITCODE），请查看 build 目录中的日志。" }
