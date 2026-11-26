"""Pruebas de los datos de entrada y de las variables externas (00_data/).

Comprueban el tamaño de los archivos, que los avisos de entrenamiento y prueba no se solapan, que las variables de
Catastro no contienen precios (evita fuga) y que las capas de Earth Engine están completas.
Si un archivo no existe, la prueba se omite.
"""
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "00_data"
PROCESADOS = DATOS / "processed"

# Tamaños documentados en docs/data.md.
N_TRAIN, N_TEST = 38_644, 10_286
N_COORDENADAS_UNICAS = 22_015
CAPAS_GEE = ["elevacion_m", "pendiente_grados", "ndvi_200m", "luces_nocturnas",
             "altura_construida_m", "poblacion_100m"]
VARIABLES_CATASTRO = ["ln_vref_manzana", "ln_vref_300m", "ln_avaluo_ph_manzana", "ln_avaluo_ph_300m",
                      "pisos_manzana", "pisos_300m", "estrato_manzana", "estrato_medio_300m",
                      "prop_estrato6_300m", "n_manzanas_300m", "dist_manzana_m"]

try:
    import pandas as pd
except ImportError:
    pd = None


def _existe(*rutas):
    return pd is not None and all(r.exists() for r in rutas)


@unittest.skipUnless(_existe(DATOS / "train.csv", DATOS / "test.csv"), "faltan train.csv o test.csv")
class Avisos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(DATOS / "train.csv", usecols=["property_id", "price"])
        cls.test = pd.read_csv(DATOS / "test.csv", usecols=lambda c: c in ("property_id", "price", "lat", "lon"))

    def test_tamanos_documentados(self):
        self.assertEqual(len(self.train), N_TRAIN)
        self.assertEqual(len(self.test), N_TEST)

    def test_entrenamiento_tiene_precio_y_prueba_no(self):
        self.assertTrue(self.train["price"].notna().all())
        self.assertNotIn("price", self.test.columns)

    def test_identificadores_unicos_y_sin_solapamiento(self):
        self.assertEqual(self.train["property_id"].nunique(), len(self.train))
        self.assertEqual(self.test["property_id"].nunique(), len(self.test))
        self.assertEqual(len(set(self.train["property_id"]) & set(self.test["property_id"])), 0)


@unittest.skipUnless(_existe(PROCESADOS / "variables_catastro.parquet", DATOS / "train.csv", DATOS / "test.csv"),
                     "faltan variables_catastro.parquet o los CSV de avisos")
class CatastroSinFuga(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = pd.read_parquet(PROCESADOS / "variables_catastro.parquet")
        cls.avisos = set(pd.read_csv(DATOS / "train.csv", usecols=["property_id"])["property_id"]) | set(
            pd.read_csv(DATOS / "test.csv", usecols=["property_id"])["property_id"])

    def test_contiene_las_once_variables(self):
        self.assertTrue(set(VARIABLES_CATASTRO).issubset(self.cat.columns))

    def test_no_contiene_precios(self):
        sospechosas = [c for c in self.cat.columns if "price" in c.lower() or "precio" in c.lower()]
        self.assertEqual(sospechosas, [], "las variables de Catastro no deben usar precios")

    def test_un_registro_por_aviso(self):
        self.assertEqual(self.cat["property_id"].nunique(), len(self.cat))

    def test_cubre_todos_los_avisos(self):
        faltantes = self.avisos - set(self.cat["property_id"])
        self.assertEqual(len(faltantes), 0, f"{len(faltantes)} avisos sin variables de Catastro")


@unittest.skipUnless(_existe(PROCESADOS / "gee_features.parquet"), "falta gee_features.parquet")
class CapasEarthEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gee = pd.read_parquet(PROCESADOS / "gee_features.parquet")

    def test_seis_capas_y_coordenadas(self):
        self.assertTrue(set(CAPAS_GEE).issubset(self.gee.columns))
        self.assertEqual(len(self.gee), N_COORDENADAS_UNICAS)

    def test_una_fila_por_coordenada(self):
        self.assertEqual(len(self.gee[["lat", "lon"]].drop_duplicates()), len(self.gee))

    def test_sin_faltantes_en_las_capas(self):
        self.assertEqual(int(self.gee[CAPAS_GEE].isna().sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
