# Construye las CUATRO salidas desde el mismo fuente:
#   arxiv/<n>/ + tar.gz  -> \anonfalse, autores y afiliaciones (arXiv)
#   tmlr/<n>.pdf         -> \anontrue,  doble ciego (envio a TMLR)
#
# NOTA SOBRE EL EMPAQUETADO ARXIV. Los paquetes van COMPLETAMENTE PLANOS (sin
# subdirectorio figures/) y en formato GNU en vez de pax: arXiv extrae el
# tarball en su propio arbol y los subdirectorios y las cabeceras extendidas
# de bsdtar son la causa clasica de "File `x.sty' not found". Plano + gnu es
# lo que arXiv digiere sin sorpresas. En los .tex copiados se reescribe
# "figures/" a "" para que las rutas sigan cuadrando.
$ErrorActionPreference = "Stop"
$P = "e:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\miuracognitive\paper"

$jobs = @(
  @{ n="cognition"; tex="main";        figs=@("fig_architecture.png","fig_hierarchy.png","fig_cliff_ladder.png") },
  @{ n="governor";  tex="governor_en"; figs=@("fig_hbp_concept.png","fig_vei_dynamics.png","fig_ood_corr.png","fig_v4.png") }
)

function Construye($j, $anon, $dest, $plano) {
  if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  if (-not $plano) { New-Item -ItemType Directory -Force -Path "$dest\figures" | Out-Null }

  Copy-Item "$P\$($j.tex).tex" $dest
  Copy-Item "$P\$($j.tex).bbl" $dest
  Copy-Item "$P\tmlr.sty"      $dest
  foreach ($f in $j.figs) {
    Copy-Item "$P\figures\$f" $(if ($plano) { $dest } else { "$dest\figures" })
  }

  $t = Get-Content "$dest\$($j.tex).tex" -Raw
  if ($anon)  { $t = $t.Replace("`n\anonfalse", "`n\anontrue") }
  if ($plano) { $t = $t.Replace("{figures/", "{") }   # rutas sin subdirectorio
  Set-Content "$dest\$($j.tex).tex" $t -Encoding utf8 -NoNewline

  if ($anon) {
    Remove-Item "$dest\$($j.tex).bbl" -Force
    Copy-Item "$P\refs.bib" $dest
    Copy-Item "$P\tmlr.bst" $dest
  }
  Push-Location $dest
  pdflatex -interaction=nonstopmode -halt-on-error "$($j.tex).tex" > $null
  if ($anon) { bibtex $($j.tex) > $null }
  pdflatex -interaction=nonstopmode -halt-on-error "$($j.tex).tex" > $null
  pdflatex -interaction=nonstopmode -halt-on-error "$($j.tex).tex" > $null
  $log = Get-Content "$($j.tex).log" -Raw
  $pg  = ([regex]::Match($log,"Output written on .*? \((\d+) pages")).Groups[1].Value
  $ci  = ([regex]::Matches($log,"Citation .*? undefined")).Count
  $nf  = ([regex]::Matches($log,"not found")).Count
  $ov  = ([regex]::Matches($log,"Overfull")).Count
  $modo = if ($anon) { "ANONIMO " } else { "con autores" }
  Write-Output "  $($j.n) $modo : $pg pag | citas indef=$ci | no hallados=$nf | overfull=$ov"
  Pop-Location
}

Write-Output "--- arXiv (autores reales, paquete PLANO) ---"
foreach ($j in $jobs) {
  $d = "$P\arxiv\$($j.n)"
  Construye $j $false $d $true
  Push-Location $d
  Remove-Item -Force *.aux,*.log,*.out,*.pdf,*.blg -ErrorAction SilentlyContinue
  Pop-Location
  $tgz = "$P\arxiv\miuracognitive-$($j.n).tar.gz"
  if (Test-Path $tgz) { Remove-Item $tgz -Force }
  # via Python: bsdtar de Windows no sabe --format=gnu y escribe pax
  & "C:\Users\fmarr\miniconda3\envs\implanto\python.exe" "$P\pack_arxiv.py" $d $tgz
}

Write-Output "--- TMLR (doble ciego) ---"
New-Item -ItemType Directory -Force -Path "$P\tmlr" | Out-Null
foreach ($j in $jobs) {
  $d = "$P\tmlr\_build_$($j.n)"
  Construye $j $true $d $false
  Copy-Item "$d\$($j.tex).pdf" "$P\tmlr\tmlr-submission-$($j.n).pdf"
  Remove-Item -Recurse -Force $d
}

Write-Output "`n--- entregables ---"
Get-ChildItem "$P\arxiv\*.tar.gz","$P\tmlr\*.pdf" | ForEach-Object {
  Write-Output ("  {0,-42} {1,9:N0} bytes" -f $_.Name, $_.Length)
}
