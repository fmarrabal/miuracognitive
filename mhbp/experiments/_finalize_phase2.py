"""Watcher de la Fase 2: espera a que mueran los workers del confirmatorio y
corre el agregador estadístico. Desacoplado. Uso: python _finalize_phase2.py <pids...>"""
import os
import subprocess
import sys
import time

PIDS = [int(x) for x in sys.argv[1:]]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "mhbp", "tasks", "synthetic_multiscale", "results_phase2")


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True)
        return str(pid) in out.stdout
    except Exception:
        return False


t0 = time.time()
while any(alive(p) for p in PIDS) and (time.time() - t0) < 30 * 3600:
    time.sleep(180)

env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
rcs = []
for mod in ("mhbp.analysis.phase2_report", "mhbp.analysis.phase2b_report"):
    r = subprocess.run([sys.executable, "-m", mod],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    rcs.append(f"{mod}: rc={r.returncode}")
with open(os.path.join(RES, "FINALIZED.txt"), "w", encoding="utf-8") as f:
    f.write(f"finalizado {time.ctime()} tras {(time.time()-t0)/3600:.2f} h\n")
    f.write("\n".join(rcs) + "\n")
