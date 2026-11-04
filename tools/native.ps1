# 原生命令的最小安全封装。
#
# 背景(2026-10-26 事故):构建脚本第一行是 $ErrorActionPreference='Stop'。PyInstaller
# 每次都会往 stderr 写警告,而在 PowerShell 7 里原生命令的 stderr 会被折成
# ErrorRecord;一旦策略是 Stop,第一条警告就把整段脚本打断。用
# `powershell -File tools/build.ps1` 从后台作业里跑时,这个中断只会留在父进程的
# 输出里,子进程返回 0,于是"门禁绿、打包没做、发布包装的是旧 exe"看起来完全成功。
#
# 这里的约定:原生命令一律走 Invoke-Native——把偏好临时降到 Continue(错误只打印,
# 不中断),显式检查退出码,并且用清空 LASTEXITCODE 的办法识别"命令根本没启动"。

function Invoke-Native {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$Name
    )
    Write-Host ("==> {0}: {1} {2}" -f $Name, $FilePath, ($Arguments -join ' '))
    $previous = $ErrorActionPreference
    # 原生命令的 stderr 不能中断流水线(见文件头注释)。
    $ErrorActionPreference = 'Continue'
    try {
        $global:LASTEXITCODE = $null
        & $FilePath @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($null -eq $code) {
        throw ("{0} 没有返回退出码,命令很可能没有真正启动:{1}" -f $Name, $FilePath)
    }
    if ($code -ne 0) {
        throw ("{0} 失败(exit {1}):{2} {3}" -f $Name, $code, $FilePath, ($Arguments -join ' '))
    }
    Write-Host ("<== {0}: ok" -f $Name)
    return $code
}

function Assert-FreshFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][datetime]$Since,
        [Parameter(Mandatory = $true)][string]$What
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw ("{0} 没有生成:{1}" -f $What, $Path)
    }
    $stamp = (Get-Item -LiteralPath $Path).LastWriteTime
    if ($stamp -lt $Since) {
        throw ("{0} 没有更新(文件时间 {1} 早于本次开始时间 {2}),上一步很可能被跳过了:{3}" -f `
            $What, $stamp.ToString('yyyy-MM-dd HH:mm:ss'), $Since.ToString('yyyy-MM-dd HH:mm:ss'), $Path)
    }
    Write-Host ("   {0}: {1} @ {2}" -f $What, (Split-Path -Leaf $Path), $stamp.ToString('MM-dd HH:mm:ss'))
}
