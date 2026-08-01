<#
  run_multimodel.ps1 — CONVENIÊNCIA (opcional) para rodar a avaliação multi-modelo
  em sequência no Windows. O CAMINHO PRINCIPAL é o fluxo MANUAL:

    Para cada modelo:
      1) suba o llama-server:   llama-server -m <modelo>.gguf --port 8080 --n-probs 5
      2) rode a avaliação:      python evaluate_local_accuracy.py --url http://127.0.0.1:8080 --model-name <rotulo>
      3) ao final dos modelos:  python aggregate_multimodel.py

  Este script apenas automatiza, para CADA modelo da lista abaixo:
    start-server -> espera /health ficar "ok" -> roda o eval -> encerra o server.
  Ao terminar todos, roda o aggregate uma vez.

  Edite $Models com (Label, Gguf) e ajuste $LlamaServerExe. Uso:
    powershell -ExecutionPolicy Bypass -File .\run_multimodel.ps1
#>

param(
  [int]    $Port           = 8080,
  [string] $LlamaServerExe = "llama-server",   # caminho do executavel do llama.cpp (ou no PATH)
  [string] $Python         = "python",
  [int]    $NProbs         = 5,
  [int]    $HealthTimeoutSec = 180,
  [switch] $SkipAggregate
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Url  = "http://127.0.0.1:$Port"

# >>> EDITE AQUI: rótulo -> caminho do GGUF de cada modelo local <<<
$Models = @(
  @{ Label = "gemma-3-1b";    Gguf = "C:\models\gemma-3-1b-it-Q4_K_M.gguf" },
  @{ Label = "qwen2.5-0.5b";  Gguf = "C:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf" },
  @{ Label = "qwen2.5-1.5b";  Gguf = "C:\models\qwen2.5-1.5b-instruct-q4_k_m.gguf" },
  @{ Label = "llama-3.2-1b";  Gguf = "C:\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" }
)

function Wait-Health {
  param([string]$Url, [int]$TimeoutSec)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -Uri "$Url/health" -UseBasicParsing -TimeoutSec 5
      if ($r.StatusCode -eq 200 -and $r.Content -match "ok") { return $true }
    } catch { Start-Sleep -Milliseconds 800 }
    Start-Sleep -Milliseconds 800
  }
  return $false
}

foreach ($m in $Models) {
  $label = $m.Label
  $gguf  = $m.Gguf
  Write-Host "==== Modelo: $label ====" -ForegroundColor Cyan

  if (-not (Test-Path $gguf)) {
    Write-Warning "GGUF nao encontrado: $gguf  — pulando '$label'."
    continue
  }

  Write-Host "Subindo llama-server (porta $Port)..."
  $srv = Start-Process -FilePath $LlamaServerExe `
      -ArgumentList @("-m", $gguf, "--port", "$Port", "--n-probs", "$NProbs") `
      -PassThru -WindowStyle Minimized

  try {
    if (-not (Wait-Health -Url $Url -TimeoutSec $HealthTimeoutSec)) {
      Write-Warning "Servidor de '$label' nao respondeu /health a tempo — pulando."
      continue
    }
    Write-Host "Servidor OK. Rodando avaliacao para '$label'..." -ForegroundColor Green
    & $Python (Join-Path $Here "evaluate_local_accuracy.py") `
        "--url" $Url "--model-name" $label
    if ($LASTEXITCODE -ne 0) { Write-Warning "Eval de '$label' retornou codigo $LASTEXITCODE." }
  }
  finally {
    Write-Host "Encerrando servidor de '$label'..."
    if ($srv -and -not $srv.HasExited) {
      Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2   # solta a porta antes do proximo modelo
  }
}

if (-not $SkipAggregate) {
  Write-Host "==== Agregando resultados ====" -ForegroundColor Cyan
  & $Python (Join-Path $Here "aggregate_multimodel.py")
}
Write-Host "Concluido." -ForegroundColor Green
