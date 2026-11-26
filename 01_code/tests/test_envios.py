"""Pruebas de los archivos de envío a Kaggle en 02_outputs/predictions/.

Revisan el formato, no el error: que cada archivo tenga las filas, los identificadores, el formato de texto y los
precios que exige la plantilla. Si un archivo no existe, la prueba se omite.
"""
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]            # raíz del proyecto sebastian/
PREDICCIONES = RAIZ / "02_outputs" / "predictions"
PLANTILLA = RAIZ / "00_data" / "submission_template.csv"
PRECIO_MIN, PRECIO_MAX = 1e8, 5e9                    # rango de sentido común en pesos (no es un límite del concurso)


def _envios():
    if not PREDICCIONES.exists():
        return []
    return sorted(p for p in PREDICCIONES.glob("*.csv") if not p.name.startswith("registro"))


try:
    import pandas as pd
except ImportError:                                  # sin pandas, las pruebas de formato se omiten
    pd = None


@unittest.skipUnless(PLANTILLA.exists() and pd is not None, "falta la plantilla o pandas")
class EnviosKaggle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plantilla = pd.read_csv(PLANTILLA)
        cls.ids = set(cls.plantilla["property_id"])

    def test_hay_al_menos_un_envio(self):
        if not _envios():
            self.skipTest("no hay archivos en 02_outputs/predictions/")

    def test_formato_crlf_y_encabezado(self):
        for ruta in _envios():
            with self.subTest(archivo=ruta.name):
                contenido = ruta.read_bytes()
                self.assertTrue(contenido.endswith(b"\r\n"), "el archivo no termina en CRLF")
                self.assertEqual(contenido.count(b"\n"), contenido.count(b"\r\n"),
                                 "hay saltos de línea LF sueltos")
                encabezado = contenido.split(b"\r\n", 1)[0].decode("utf-8").split(",")
                self.assertEqual(encabezado, ["property_id", "price"])

    def test_filas_e_identificadores_coinciden_con_la_plantilla(self):
        for ruta in _envios():
            with self.subTest(archivo=ruta.name):
                df = pd.read_csv(ruta)
                self.assertEqual(len(df), len(self.plantilla))
                self.assertEqual(df["property_id"].nunique(), len(df), "identificadores repetidos")
                self.assertEqual(set(df["property_id"]), self.ids)

    def test_precios_enteros_y_en_rango(self):
        for ruta in _envios():
            with self.subTest(archivo=ruta.name):
                precio = pd.read_csv(ruta)["price"]
                self.assertTrue(precio.notna().all(), "hay precios vacíos")
                self.assertTrue((precio == precio.round(0)).all(), "hay precios con decimales")
                self.assertTrue(((precio >= PRECIO_MIN) & (precio <= PRECIO_MAX)).all(),
                                f"hay precios fuera de [{PRECIO_MIN:.0e}, {PRECIO_MAX:.0e}]")


if __name__ == "__main__":
    unittest.main()
