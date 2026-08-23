"""Watcher: espera a que el barrido nocturno (PID dado) termine, luego corre el
agregador para dejar REPORT.md + figuras listos por la mañana. Desacoplado."""
import os, sys, time, subprocess

PID = int(sys.argv[1]) if len(sys.argv) > 1 else None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "scale_study", "results")
PY = sys.executable


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True)
        return str(pid) in out.stdout
    except Exception:
        return False


t0 = time.time()
# espera a que muera el barrido (o 11 h de guarda)
while PID and alive(PID) and (time.time() - t0) < 11 * 3600:
    time.sleep(120)

# agrega dos veces por si acaso (una intermedia, otra final tras breve margen)
env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
for _ in range(1):
    subprocess.run([PY, "-m", "scale_study.aggregate"], cwd=ROOT, env=env,
                   capture_output=True, text=True)
with open(os.path.join(RES, "FINALIZED.txt"), "w", encoding="utf-8") as f:
    f.write(f"finalizado {time.ctime()} tras {(time.time()-t0)/3600:.2f} h de espera\n")
