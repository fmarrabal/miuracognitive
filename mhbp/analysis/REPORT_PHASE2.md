# Fase 2 — Resultados del confirmatorio (SMA)

Corridas cargadas: **184**. Generado por `phase2_report.py`.

## Métrica primaria (J en suite OOD; menor = mejor) y secundarias

| controlador | n | J_OOD | regret_OOD | viol_rate(iid) | J_iid | params(core/total) |
|---|---|---|---|---|---|---|
| ar2 (explor.) | 8 | 2.2652±0.3024 | 1.2521 | 0.050 | 1.8075 | 5060/5412 |
| gru | 20 | 2.2115±0.2310 | 1.1984 | 0.046 | 1.7945 | 8160/8480 |
| gru_gov (explor.) | 20 | 2.0502±0.2019 | 1.0371 | 0.026 | 1.7685 | 11296/11296 |
| hbp_single (explor.) | 8 | 3.5373±0.2755 | 2.5242 | 0.077 | 2.1506 | 196/10852 |
| mhbp | 24 | 10.1104±1.7523 | 9.0973 | 0.024 | 1.8705 | 3299/8307 |
| mhbp_first | 24 | 7.9794±1.1780 | 6.9663 | 0.043 | 1.8787 | 3299/8307 |
| mhbp_gov (explor.) | 20 | 2.3206±0.4244 | 1.3075 | 0.024 | 1.8332 | 6531/11539 |
| mhbp_noallo (explor.) | 8 | 10.4492±1.2654 | 9.4361 | 0.017 | 1.9045 | 387/8307 |
| mhbp_nocpl (explor.) | 8 | 10.6298±1.2267 | 9.6167 | 0.023 | 1.9556 | 3168/8176 |
| mhbp_taueq | 16 | 6.0090±0.5698 | 4.9959 | 0.022 | 1.8427 | 3299/8307 |
| mlp (explor.) | 8 | 1.9064±0.1029 | 0.8933 | 0.022 | 1.7412 | 7248/7632 |
| react (explor.) | 20 | 2.0524±0.1260 | 1.0392 | 0.029 | 1.7936 | 2976/2976 |

## Contrastes primarios (pareados por semilla; Holm-3 sobre p_t bilateral,
Wilcoxon como sensibilidad; Δ<0 ⇒ mhbp mejor; 'Holm SÍ' EXIGE Δ<0)

| contraste | n | Δ(J_OOD) | t | p | p_wilcoxon | dz | IC95 | Holm |
|---|---|---|---|---|---|---|---|---|
| C1_mhbp_vs_gru | 20 | +7.6792 | 18.17 | 0.0000 | 0.0000 | 4.06 | [+6.9176, +8.5218] | dir. contraria |
| C2_mhbp_vs_taueq | 16 | +3.9312 | 8.95 | 0.0000 | 0.0000 | 2.24 | [+3.1675, +4.7796] | dir. contraria |
| C3_mhbp_vs_first | 24 | +2.1310 | 4.62 | 0.0001 | 0.0000 | 0.94 | [+1.2108, +3.0133] | dir. contraria |
| X4_mhbp_vs_single | 8 | +6.5625 | 9.82 | 0.0000 | 0.0078 | 3.47 | [+5.4555, +7.8684] | (explor.) |
| X5_mhbp_vs_nocpl | 8 | -0.5301 | -0.59 | 0.5721 | 0.6406 | -0.21 | [-2.1836, +1.0132] | (explor.) |
| X6_mhbp_vs_noallo | 8 | -0.3494 | -0.44 | 0.6736 | 0.8438 | -0.16 | [-1.7547, +1.0903] | (explor.) |

Sensibilidad de la métrica: la variante 'media plana de 4 protocolos' (ood_flat4_J) se reporta aparte y NO entra en Holm.

## J por protocolo (media entre semillas)

| controlador | iid | ood_budget_lo | ood_budget_hi | ood_riskfreq | ood_long | e4_step | e2_risk_only |
|---|---|---|---|---|---|---|---|
| ar2 | 1.808 | 2.989 | 3.414 | 1.775 | 1.819 | 4.288 | 2.080 |
| gru | 1.795 | 3.063 | 3.091 | 1.764 | 1.793 | 4.501 | 2.071 |
| gru_gov | 1.769 | 2.708 | 2.510 | 1.758 | 1.784 | 3.241 | 2.085 |
| hbp_single | 2.151 | 4.835 | 7.890 | 2.118 | 2.132 | 6.548 | 2.372 |
| mhbp | 1.871 | 16.646 | 35.947 | 1.877 | 2.158 | 23.205 | 2.139 |
| mhbp_first | 1.879 | 7.235 | 32.950 | 1.868 | 1.978 | 8.547 | 2.092 |
| mhbp_gov | 1.833 | 2.605 | 3.746 | 1.819 | 1.967 | 2.801 | 2.154 |
| mhbp_noallo | 1.904 | 18.604 | 35.502 | 1.911 | 2.383 | 24.858 | 2.164 |
| mhbp_nocpl | 1.956 | 13.343 | 40.616 | 1.981 | 2.929 | 20.320 | 2.271 |
| mhbp_taueq | 1.843 | 6.552 | 22.116 | 1.837 | 1.856 | 12.021 | 2.210 |
| mlp | 1.741 | 2.744 | 1.700 | 1.739 | 1.758 | 3.108 | 1.981 |
| react | 1.794 | 2.718 | 2.393 | 1.787 | 1.815 | 2.909 | 2.098 |

## Seguimiento (correlaciones acción↔factor, protocolo iid) y E4

| controlador | corr(d,depth)↑ | corr(ρ,tool)↑ | corr(d,halt)↓ | corr(B,gasto)↑ | settling(vent.) | viol. post-escalón |
|---|---|---|---|---|---|---|
| ar2 | — | 0.038 | -0.129 | 0.908 | 0.705 | 9.110 |
| gru | — | 0.054 | -0.109 | 0.918 | 0.785 | 9.663 |
| gru_gov | — | 0.001 | -0.101 | 0.932 | 0.478 | 5.844 |
| hbp_single | — | 0.011 | -0.007 | 0.929 | 1.470 | 15.725 |
| mhbp | — | 0.007 | 0.007 | 0.931 | 2.523 | 37.272 |
| mhbp_first | — | 0.009 | 0.002 | 0.935 | 1.270 | 18.482 |
| mhbp_gov | — | -0.003 | -0.075 | 0.933 | 0.276 | 3.777 |
| mhbp_noallo | — | 0.005 | 0.006 | 0.937 | 2.425 | 38.619 |
| mhbp_nocpl | — | 0.008 | 0.007 | 0.937 | 2.110 | 33.946 |
| mhbp_taueq | — | 0.003 | 0.006 | 0.941 | 2.358 | 26.206 |
| mlp | — | 0.005 | -0.142 | 0.939 | 0.415 | 5.260 |
| react | — | 0.001 | -0.155 | 0.937 | 0.381 | 4.466 |

Figuras: `analysis/figures_phase2/phase2_primary.png`
