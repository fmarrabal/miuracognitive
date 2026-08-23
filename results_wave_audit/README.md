# Auditoría de la ecuación de onda

Estado: **PASS**.

La rama `hbp_full` implementa, para `u=h-h*`, la recurrencia

```text
u[n+1] = 2u[n] - u[n-1]
         - dt² (ω₀² I + c² L) u[n]
         - 2 ζω₀ dt (u[n] - u[n-1])
         + dt² F[n]
```

Es un Verlet de posición con velocidad atrasada para el amortiguamiento. `L`
es el laplaciano combinatorio de la cadena, equivalente a una frontera natural
de grafo: simétrico, semidefinido positivo y con suma de filas cero.

## Comprobaciones independientes

| prueba | resultado |
|---|---:|
| error máximo del espectro analítico de `L` | `4,77e-7` |
| error máximo contra la recurrencia modal cerrada | `7,45e-8` |
| propagación espacial | exactamente un enlace por tick |
| crecimiento libre máximo | ninguno (`max‖u‖/‖u₀‖=1`) |
| decaimiento tras 120 ticks | `5,02e-19` |
| radio espectral por defecto | `0,70711` |
| radio al forzar `dt=3`, fuera de CFL | `7,19366` |
| penalización fuera de CFL | `30,03369` |

El test de carga confirma además que la modulación depende del input, `ω₀`,
`ζ`, `c` y la ganancia de autocheck reciben gradiente, `ω₀/ζ` cambian durante
optimización y las ramas de primer y segundo orden no son idénticas.

## Checkpoints reales

Se reconstruyeron tres checkpoints de `results_budgeted_stream` y uno de AHA.
Todos tienen penalización de estabilidad cero. Sus radios espectrales exactos
son `0,91323`, `0,89354`, `0,90115` y `0,73980`, estrictamente menores que uno.

Conclusión: la ecuación, sus signos, la discretización espacial, la recurrencia
temporal y el certificado están funcionando de forma coherente. Esto valida la
dinámica numérica; no prueba por sí solo que la onda sea la causa de una mejora
cognitiva, cuestión que los contrastes del zoo ya separan.

Archivos:

- `audit.json`: métricas, criterios y checkpoints;
- `wave_equation_audit.py`: auditoría reproducible;
- `tests/test_hbp_wave.py`: seis regresiones unitarias.
