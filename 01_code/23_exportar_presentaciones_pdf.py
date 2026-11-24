"""23 · Exporta las presentaciones a PDF con PowerPoint (paso opcional; solo Windows con Microsoft PowerPoint instalado).

Qué hace
    Abre 03_slides/best_equipo_02.pptx y 03_slides/compare_equipo_02.pptx con PowerPoint por COM (PowerShell) y las
    guarda como PDF en la misma carpeta, con el mismo nombre. Si PowerPoint ya estaba abierto, no lo cierra.

Cuándo corre
    Solo si se pide con la variable de entorno PS3_EXPORTAR_PDF=1. Sin ella, o si no hay Windows o PowerPoint, el script
    avisa y termina sin error, para que el script maestro no falle. Los PDF finales ya están en 03_slides/.
"""
from functions.setup_python import *  # noqa: F401,F403  (raíz del repositorio, semilla y carpetas DIR_*)

import os
import platform
import subprocess
import sys

from functions.entorno_ps3 import log
from functions.laminas_entorno import SALIDA_PPT

if os.environ.get("PS3_EXPORTAR_PDF", "0") != "1":
    log("Exportación a PDF omitida (es opcional: PS3_EXPORTAR_PDF=1, en Windows con PowerPoint)")
    sys.exit(0)
if platform.system() != "Windows":
    log("Exportación a PDF omitida: requiere Windows con Microsoft PowerPoint")
    sys.exit(0)

for nombre in ("best_equipo_02", "compare_equipo_02"):
    pptx = (SALIDA_PPT / f"{nombre}.pptx").resolve()
    pdf = pptx.with_suffix(".pdf")
    if not pptx.exists():
        log(f"No existe {pptx.name}: corra antes los scripts 21 y 22")
        continue
    orden = (f"$antes = Get-Process POWERPNT -ErrorAction SilentlyContinue; "
             f"$app = New-Object -ComObject PowerPoint.Application; "
             f"$p = $app.Presentations.Open('{pptx}', $true, $false, $false); "
             f"$p.SaveAs('{pdf}', 32); $p.Close(); if (-not $antes) {{ $app.Quit() }}")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", orden], check=True, timeout=600)
        log(f"PDF: {pdf.name}")
    except (OSError, subprocess.SubprocessError) as e:      # sin PowerPoint o sin PowerShell: se avisa y se sigue
        log(f"No se pudo exportar {pptx.name} a PDF ({type(e).__name__}); el resto del pipeline no se afecta")
