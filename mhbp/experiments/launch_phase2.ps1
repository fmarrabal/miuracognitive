# Fase 2 — lanza K workers desacoplados del confirmatorio (resumible).
# Uso:  powershell -File mhbp/experiments/launch_phase2.ps1
$ROOT = "e:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\miuracognitive"
$PY = "C:\Users\fmarr\miniconda3\envs\implanto\python.exe"
$RES = "$ROOT\mhbp\tasks\synthetic_multiscale\results_phase2"
New-Item -ItemType Directory -Force $RES | Out-Null
$env:PYTHONPATH = $ROOT
$env:PYTHONIOENCODING = "utf-8"
$K = 6
$STEPS = 500
# 1) prewarm de la cache del oraculo en un solo proceso (evita carreras)
& $PY -m mhbp.experiments.exp_phase2 --prewarm *> "$RES\prewarm.log"
# 2) workers desacoplados
$pids = @()
for ($i = 0; $i -lt $K; $i++) {
    $p = Start-Process -FilePath $PY `
        -ArgumentList "-m", "mhbp.experiments.exp_phase2", "--worker", "$i", "--steps", "$STEPS" `
        -WorkingDirectory $ROOT -WindowStyle Hidden `
        -RedirectStandardOutput "$RES\worker$i.log" `
        -RedirectStandardError "$RES\worker$i.err" -PassThru
    $pids += $p.Id
    Start-Sleep -Seconds 3
}
"workers lanzados: $($pids -join ', ')" | Tee-Object "$RES\workers.pids"
