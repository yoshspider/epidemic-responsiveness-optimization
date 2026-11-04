param(
    [string]$Module = 'epidemic.moderate_ideal_confirm',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$SimulationArgs
)

$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $projectPython = $pythonCommand.Source
}

Push-Location $PSScriptRoot
try {
    & $projectPython -m $Module @SimulationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
