"""Empaqueta un directorio para arXiv: TODO PLANO y en formato GNU.

Por que no se usa `tar` del sistema: en Windows es bsdtar, que escribe
cabeceras pax extendidas y no acepta --format=gnu. Las cabeceras extendidas y
los subdirectorios son la causa clasica del error de arXiv
"File `x.sty' not found": su extractor no siempre reconstruye el arbol como
uno espera. Con GNU_FORMAT y ficheros en la raiz no hay ambiguedad posible.

Ademas se normalizan uid/gid/mtime y permisos, para que el tarball sea
reproducible y no lleve rastros del sistema de ficheros de Windows.

  python pack_arxiv.py <directorio> <salida.tar.gz>
"""
import os
import sys
import tarfile


def empaqueta(origen, salida):
    ficheros = sorted(
        f for f in os.listdir(origen)
        if os.path.isfile(os.path.join(origen, f))
    )
    if not ficheros:
        raise SystemExit(f"nada que empaquetar en {origen}")
    if any(os.path.isdir(os.path.join(origen, d)) for d in os.listdir(origen)):
        raise SystemExit(f"{origen} tiene subdirectorios: el paquete de arXiv "
                         f"debe ir PLANO")

    def limpia(ti):
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = 1600000000          # fijo: tarball reproducible
        ti.mode = 0o644
        return ti

    with tarfile.open(salida, "w:gz", format=tarfile.GNU_FORMAT) as t:
        for f in ficheros:
            t.add(os.path.join(origen, f), arcname=f, filter=limpia)
    return ficheros


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    fs = empaqueta(sys.argv[1], sys.argv[2])
    # el formato se lee de la CABECERA: tarfile.open() en lectura devuelve
    # siempre su valor por defecto (PAX) y no el del fichero
    import gzip
    with gzip.open(sys.argv[2], "rb") as g:
        magic = g.read(512)[257:265]
    fmt = ("GNU" if magic == b"ustar  \x00"
           else "USTAR/POSIX" if magic[:5] == b"ustar" else "?")
    with tarfile.open(sys.argv[2]) as t:
        nombres = t.getnames()
    print(f"{os.path.basename(sys.argv[2])}: {len(nombres)} ficheros, "
          f"formato {fmt}")
    for n in nombres:
        print(f"    {n}")
    hondos = [n for n in nombres if "/" in n]
    if hondos:
        raise SystemExit(f"ERROR: hay rutas con subdirectorio: {hondos}")
