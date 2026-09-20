[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$SubjectSaltFile,

    [ValidateRange(1, 64)]
    [int]$AuditWorkers = 8,

    [ValidateRange(1, 64)]
    [int]$PreprocessWorkers = 8,

    [ValidateRange(1, 64)]
    [int]$FeatureWorkers = 8,

    [switch]$SkipRuff,

    [switch]$SkipFusion
)

Write-Warning "Deprecated: use docs/clean_retrain_runbook_2026-09-20.md and run each gated step explicitly."

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "`n== $Name ==" -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$cleanRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$saltPath = [System.IO.Path]::GetFullPath($SubjectSaltFile)

if ($cleanRoot -eq [System.IO.Path]::GetPathRoot($cleanRoot)) {
    throw "OutputRoot cannot be a drive root"
}
if (Test-Path -LiteralPath $cleanRoot) {
    throw "OutputRoot already exists and will not be reused: $cleanRoot"
}
if (-not (Test-Path -LiteralPath $saltPath -PathType Leaf)) {
    throw "Subject salt file does not exist"
}
$saltText = (Get-Content -LiteralPath $saltPath -Raw).Trim()
if ($saltText -notmatch '^[0-9a-fA-F]{64}$') {
    throw "Subject salt file must contain exactly 32 bytes encoded as hexadecimal"
}

Push-Location $projectRoot
try {
    if ($env:CONDA_DEFAULT_ENV -ne "bme-model") {
        throw "Activate the bme-model conda environment before starting"
    }

    $gitStatus = @(& git status --porcelain)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the Git working tree"
    }
    if ($gitStatus.Count -gt 0) {
        throw "Git working tree is not clean; commit and sync the retrain code first"
    }
    $codeCommit = ((& git rev-parse HEAD) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $codeCommit -notmatch '^[0-9a-fA-F]{40}$') {
        throw "Could not resolve the clean Git commit"
    }

    Invoke-PythonStep "Environment check" @("scripts/check_environment.py")
    Invoke-PythonStep "Full unit test suite" @("-m", "pytest", "-q")
    if ($SkipRuff) {
        Write-Warning "Ruff was explicitly skipped; this run has not passed the static-analysis gate."
    }
    else {
        Invoke-PythonStep "Ruff" @("-m", "ruff", "check", ".")
    }
    Invoke-PythonStep "CUDA model smoke test" @(
        "scripts/smoke_test_model.py",
        "--config", "configs/dtp_fusion.yaml",
        "--batch-size", "1"
    )

    New-Item -ItemType Directory -Path $cleanRoot | Out-Null
    $privateRoot = Join-Path $cleanRoot "private"
    New-Item -ItemType Directory -Path $privateRoot | Out-Null
    Copy-Item -LiteralPath $saltPath -Destination (Join-Path $privateRoot "subject_salt.hex")
    $env:BME_OUTPUT_ROOT = $cleanRoot

    $preflightJson = @{
        code_commit = $codeCommit
        ruff_skipped = [bool]$SkipRuff
        unit_tests_required = $true
        cuda_smoke_test_required = $true
    } | ConvertTo-Json
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        (Join-Path $cleanRoot "clean_retrain_preflight.json"),
        $preflightJson,
        $utf8WithoutBom
    )

    Write-Host "Clean output root: $cleanRoot"
    Write-Host "Training code commit: $codeCommit"
    Write-Host "Only the private subject salt is reused; all derived artifacts are rebuilt."

    Invoke-PythonStep "Full schema audit" @(
        "scripts/audit_data.py",
        "--config", "configs/base.yaml",
        "--schema-zips", "all",
        "--maximum-rows", "1000000000",
        "--workers", "$AuditWorkers",
        "--no-resume"
    )

    Write-Host "`n== Repeated-header audit ==" -ForegroundColor Cyan
    & python scripts/audit_multisection.py `
        --config configs/base.yaml `
        --workers $AuditWorkers
    $multisectionExit = $LASTEXITCODE
    $multisectionReport = Join-Path $cleanRoot "v2\indices\multisection_audit.json"
    if (-not (Test-Path -LiteralPath $multisectionReport -PathType Leaf)) {
        throw "Repeated-header audit did not produce its complete report"
    }
    if ($multisectionExit -notin @(0, 1)) {
        throw "Repeated-header audit failed with unexpected exit code $multisectionExit"
    }
    if ($multisectionExit -eq 1) {
        Write-Host "Expected quarantine classifications were reported; preprocessing will verify exact counts."
    }

    Invoke-PythonStep "Full preprocessing" @(
        "scripts/preprocess_data.py",
        "--config", "configs/base.yaml",
        "--workers", "$PreprocessWorkers",
        "--overwrite"
    )
    Invoke-PythonStep "Unfrozen quality validation" @(
        "scripts/validate_data.py",
        "--config", "configs/base.yaml"
    )

    $confirmation = Read-Host "Review the anonymous quality report above. Type FREEZE to accept it"
    if ($confirmation -cne "FREEZE") {
        throw "Quality snapshot was not approved; training is blocked"
    }
    Invoke-PythonStep "Freeze quality snapshot" @(
        "scripts/validate_data.py",
        "--config", "configs/base.yaml",
        "--write-expectations"
    )
    Invoke-PythonStep "Frozen quality gate" @(
        "scripts/validate_data.py",
        "--config", "configs/base.yaml"
    )

    Invoke-PythonStep "Build baseline features" @(
        "scripts/build_features.py",
        "--config", "configs/baseline.yaml",
        "--workers", "$FeatureWorkers"
    )
    foreach ($fold in 0..4) {
        Invoke-PythonStep "Train baseline fold $fold" @(
            "scripts/train_xgboost.py",
            "--config", "configs/baseline.yaml",
            "--fold", "$fold",
            "--no-resume"
        )
    }
    Invoke-PythonStep "Summarize clean baseline" @(
        "scripts/summarize_results.py",
        "--config", "configs/baseline.yaml",
        "--experiments", "baseline"
    )

    if (-not $SkipFusion) {
        $fusionRun = "baseline_dtp_fusion_clean"
        Invoke-PythonStep "Train fusion fold 0" @(
            "scripts/train_fusion.py",
            "--config", "configs/dtp_fusion.yaml",
            "--run-name", $fusionRun,
            "--fold", "0",
            "--fresh",
            "--baseline-source-commit", $codeCommit,
            "--require-clean-baseline"
        )
        foreach ($fold in 1..4) {
            Invoke-PythonStep "Train fusion fold $fold" @(
                "scripts/train_fusion.py",
                "--config", "configs/dtp_fusion.yaml",
                "--run-name", $fusionRun,
                "--fold", "$fold",
                "--baseline-source-commit", $codeCommit,
                "--require-clean-baseline"
            )
        }
        Invoke-PythonStep "Apply five-fold fusion promotion gate" @(
            "scripts/compare_models.py",
            "--baseline", (Join-Path $cleanRoot "v2\experiments\baseline"),
            "--candidate", (Join-Path $cleanRoot "v2\experiments\$fusionRun"),
            "--output", (Join-Path $cleanRoot "v2\experiments\${fusionRun}_promotion.json")
        )
    }

    Write-Host "`nClean retrain completed: $cleanRoot" -ForegroundColor Green
    Write-Host "All scores are local evaluations until the official test interface is confirmed."
}
finally {
    $saltText = $null
    Pop-Location
}
