param([string]$OutputRoot = $env:BME_OUTPUT_ROOT)

if ($env:OS -ne 'Windows_NT') {
    throw 'This helper requires Windows.'
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $bmeDotenvPath = Join-Path (Split-Path $PSScriptRoot -Parent) '.env'
    $bmeOutputLine = Get-Content -LiteralPath $bmeDotenvPath -Encoding UTF8 |
        Where-Object { $_ -match '^\s*BME_OUTPUT_ROOT\s*=' } |
        Select-Object -Last 1
    if (-not $bmeOutputLine) {
        throw 'BME_OUTPUT_ROOT is missing from the environment and project .env.'
    }
    $OutputRoot = ($bmeOutputLine -split '=', 2)[1].Trim().Trim([char[]]@('"', "'"))
}

$bmeLongPathRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $bmeLongPathRoot.StartsWith('\\?\')) {
    if ($bmeLongPathRoot.StartsWith('\\')) {
        $bmeLongPathRoot = '\\?\UNC\' + $bmeLongPathRoot.Substring(2)
    } else {
        $bmeLongPathRoot = '\\?\' + $bmeLongPathRoot
    }
}
if (-not [System.IO.Directory]::Exists($bmeLongPathRoot)) {
    throw "Output root does not exist: $bmeLongPathRoot"
}

$env:BME_OUTPUT_ROOT = $bmeLongPathRoot
Write-Host "BME_OUTPUT_ROOT=$env:BME_OUTPUT_ROOT"
