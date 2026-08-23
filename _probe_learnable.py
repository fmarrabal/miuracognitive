"""Prueba de aprendibilidad. ¿Aprende ALGO por encima de chance?
(1) sanity: el motor nuevo sigue aprendiendo la tarea de recall trivial (debe ~1.0).
(2) RUNSUM fácil: paridad (mod=2) de pocos bits con pocos distractores.
Si (2) no despega de 0.5, RUNSUM no es aprendible a esta escala -> pivotar.
"""
from training.config import TrainConfig
from training.trainer import train_run


def show(tag, cfg):
    print(f"\n##### PROBE: {tag} #####", flush=True)
    res = train_run(cfg, results_dir=None, verbose=False)
    ec = res["eval_curve"]
    for i, s in enumerate(ec["step"]):
        print(f"  step {s:5d} corto={ec['corto'][i]:.3f} medio={ec['medio'][i]:.3f} largo={ec['largo'][i]:.3f}", flush=True)
    print(f"  FINAL: {res['final_acc']}", flush=True)


show("recall (sanity, chance~0.02, debe ~1.0)",
     TrainConfig(task="recall", variant="vanilla", max_steps=1500, max_distractors=30, seed=0))
show("runsum mod=2 writes<=4 pocos distractores (chance=0.5)",
     TrainConfig(task="runsum", variant="vanilla", mod=2, min_writes=2, max_writes=4,
                 n_distractors=8, max_steps=2500, seed=0))
show("runsum mod=2 writes<=4 SIN distractores (chance=0.5)",
     TrainConfig(task="runsum", variant="vanilla", mod=2, min_writes=2, max_writes=4,
                 n_distractors=1, max_steps=2500, seed=0))
