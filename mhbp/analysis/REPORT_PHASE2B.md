# Fase 2b — Campo-modulador vs campo-fuente (PREREG v2)

Huellas de entorno presentes: {'SIN_FP(fase2)': 124, '847a46a1d55e': 60} — las corridas 'SIN_FP' son de la Fase 2 (anteriores al fingerprint); la identidad del entorno entre fases está garantizada por registro de sesión (sin ediciones de env.py entre el confirmatorio F2 y el F2b), no por hash.

- **mhbp_gov** (n=20): J_OOD=2.3206±0.4244  J_iid=1.8332  core=6531  total=11539
- **gru_gov** (n=20): J_OOD=2.0502±0.2019  J_iid=1.7685  core=11296  total=11296
- **react** (n=20): J_OOD=2.0524±0.1260  J_iid=1.7936  core=2976  total=2976
- **mhbp** (n=24): J_OOD=10.1104±1.7523  J_iid=1.8705  core=3299  total=8307
- **gru** (n=20): J_OOD=2.2115±0.2310  J_iid=1.7945  core=8160  total=8480
- **mlp** (n=8): J_OOD=1.9064±0.1029  J_iid=1.7412  core=7248  total=7632

## Familia Holm-3 (pareado por semilla; dirección EXPLÍCITA)

| contraste | n | Δ(J_OOD) | dir. | t | p | p_wx | dz | IC95 | Holm | nota |
|---|---|---|---|---|---|---|---|---|---|---|
| D1_gov_vs_mhbp | 20 | -7.5702 | gov mejor | -19.35 | 0.0000 | 0.0000 | -4.33 | [-8.3655, -6.8687] | SÍ | esperado gov MEJOR (mecanismo) |
| D2b_gov_vs_grugov | 20 | +0.2704 | gov PEOR | 3.37 | 0.0032 | 0.0010 | 0.75 | [+0.1236, +0.4339] | SÍ | física del campo aislada |
| D3_gov_vs_gru | 20 | +0.1091 | gov PEOR | 0.97 | 0.3466 | 0.8124 | 0.22 | [-0.0971, +0.3415] | no | contexto |

## Contrastes etiquetados (fuera de Holm)

| contraste | n | Δ(J_OOD) | dir. | t | p | p_wx | dz | IC95 | Holm | nota |
|---|---|---|---|---|---|---|---|---|---|---|
| D2a_gov_vs_react | 20 | +0.2683 | gov PEOR | 2.75 | 0.0128 | 0.0328 | 0.61 | [+0.0947, +0.4696] | (fuera) | CONFUNDIDO: modulación+memoria+capacidad |
| E1_react_vs_mlp | 8 | +0.1473 | react peor | 4.29 | 0.0036 | 0.0078 | 1.52 | [+0.0862, +0.2140] | (fuera) | gate de equivalencia del reactivo |

## Mecanismos (términos del fallo de la F2; umbrales pre-registrados:
plan·λ ≤ 1.0 y hard·λ ≤ 5.0 = 'colapsa a baseline')

| controlador | plan(budget_hi)·λ | hard(e4)·λ | settling(vent.) |
|---|---|---|---|
| mhbp | 34.54 | 21.11 | 2.52 |
| mhbp_gov | 2.30 | 0.68 | 0.28 |
| gru_gov | 1.11 | 1.15 | 0.48 |
| react | 0.93 | 0.79 | 0.38 |
| gru | 1.69 | 2.41 | 0.78 |
| mlp | 0.29 | 1.03 | 0.42 |

## Intervención g≡1 (¿la modulación es load-bearing?)

| controlador | J_OOD (normal) | J_OOD (g≡1) | Δ | lectura |
|---|---|---|---|---|
| mhbp_gov | 2.3206 | 2.3206 | +0.0000 | inerte |
| gru_gov | 2.0502 | 2.0502 | +0.0000 | inerte |
