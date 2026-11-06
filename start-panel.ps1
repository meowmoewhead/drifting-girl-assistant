param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskExe = Join-Path $taskRoot 'mumu-panel.exe'
$taskCommand = if ($NoBrowser) { 'serve' } else { 'desktop' }
try {
    if ($NoBrowser) {
        try {
            $taskReply = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 2
            if ($null -ne $taskReply.features) { return }
        } catch { }
    }
    if (Test-Path -LiteralPath $taskExe -PathType Leaf) {
        Start-Process -FilePath $taskExe -ArgumentList $taskCommand -WorkingDirectory $taskRoot -WindowStyle Hidden
    } else {
        $taskPython = Join-Path $taskRoot '.venv/Scripts/pythonw.exe'
        if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw '未找到程序，请完整解压 Windows 发行包。' }
        $taskArguments = '"' + (Join-Path $taskRoot 'desktop_entry.py') + '" ' + $taskCommand
        Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden
    }
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, '漂流助手') | Out-Null
    exit 1
}
