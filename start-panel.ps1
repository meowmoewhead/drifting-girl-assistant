param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskConfig = Join-Path $taskRoot 'config.json'
$taskExample = Join-Path $taskRoot 'config.example.json'
if (-not (Test-Path -LiteralPath $taskConfig) -and (Test-Path -LiteralPath $taskExample)) {
    Copy-Item -LiteralPath $taskExample -Destination $taskConfig -ErrorAction Stop
}
$taskUrl = 'http://127.0.0.1:8765/'

function Get-PanelState {
    try {
        $taskReply = Invoke-RestMethod -Uri ($taskUrl + 'api/status') -TimeoutSec 2
        if ($null -ne $taskReply.features -and $null -ne $taskReply.rules) { return $taskReply }
    } catch { }
    return $null
}

try {
    $taskState = Get-PanelState
    if ($null -eq $taskState) {
        $taskLog = Join-Path $taskRoot 'logs'
        New-Item -ItemType Directory -Path $taskLog -Force | Out-Null
        $taskExe = Join-Path $taskRoot 'mumu-panel.exe'
        if (Test-Path -LiteralPath $taskExe -PathType Leaf) {
            $taskProcess = Start-Process -FilePath $taskExe -ArgumentList 'serve' -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLog 'service.out.log') -RedirectStandardError (Join-Path $taskLog 'service.err.log') -PassThru
        } else {
            $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
            if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Panel executable and Python environment are both missing.' }
            $taskArguments = '"' + (Join-Path $taskRoot 'app.py') + '" serve'
            $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLog 'service.out.log') -RedirectStandardError (Join-Path $taskLog 'service.err.log') -PassThru
        }
        for ($taskAttempt = 0; $taskAttempt -lt 20; $taskAttempt++) {
            Start-Sleep -Milliseconds 500
            $taskState = Get-PanelState
            if ($null -ne $taskState) { break }
            if ($taskProcess.HasExited) { throw 'Panel service exited. See logs/service.err.log.' }
        }
        if ($null -eq $taskState) { throw 'Panel service did not become ready. See logs/service.err.log.' }
    }
    if (-not $NoBrowser) { Start-Process -FilePath $taskUrl }
    Write-Output ('Panel ready: ' + $taskUrl)
    Write-Output ('Features: ' + $taskState.features.Count + '; automation running: ' + $taskState.running)
} catch {
    if (-not $NoBrowser) {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Drifting Girl Assistant') | Out-Null
    }
    Write-Error $_
    exit 1
}
