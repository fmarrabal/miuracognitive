import re

tex = open("paper.tex", encoding="utf-8").read()
bib = open("refs.bib", encoding="utf-8").read()

cites = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        cites.add(k.strip())
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))

missing = sorted(cites - bibkeys)
unused = sorted(bibkeys - cites)
print(f"citas distintas en paper: {len(cites)} | entradas en refs.bib: {len(bibkeys)}")
print("FALTAN en refs.bib (referencias indefinidas):", missing or "ninguna  OK")
print("sin usar en refs.bib:", unused or "ninguna")

# balance de entornos
envs = {}
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
    envs[m.group(2)] = envs.get(m.group(2), 0) + (1 if m.group(1) == "begin" else -1)
unbal = {k: v for k, v in envs.items() if v != 0}
print("entornos desbalanceados:", unbal or "ninguno  OK")

# llaves
print("balance de llaves { }:", tex.count("{") - tex.count("}"), "(0 = OK)")
print("secciones:", re.findall(r"\\section\{([^}]+)\}", tex))
