"""Ejecuta todas las pruebas de 01_code/tests/.

Uso (desde la raíz del proyecto), sin dependencias nuevas: solo usa unittest de la biblioteca estándar.

    python 01_code/tests/run_tests.py            # todas las pruebas
    python 01_code/tests/run_tests.py -v         # con el detalle de cada prueba

En este entorno, los scripts del proyecto se ejecutan con el ejecutor de la política (agents_policy/hooks/runtime.py);
las pruebas siguen el mismo criterio: correrlas requiere aprobación del alcance antes.

Las pruebas de particiones usan datos sintéticos. Las de datos y envíos leen 00_data/ y 02_outputs/predictions/ en
modo lectura y se omiten si falta algún archivo.

Código de salida: 0 si todas pasan, 1 si alguna falla o hay errores.
"""
import sys
import unittest
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def main(argv=None):
    verbose = "-v" in (argv if argv is not None else sys.argv[1:])
    suite = unittest.defaultTestLoader.discover(start_dir=str(AQUI), pattern="test_*.py",
                                                top_level_dir=str(AQUI))
    resultado = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    return 0 if resultado.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
