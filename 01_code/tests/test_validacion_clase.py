"""Pruebas de las particiones de validación cruzada espacial (functions/validacion_clase.py).

Usan datos sintéticos pequeños: no leen el proyecto ni necesitan internet.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

CODIGO = Path(__file__).resolve().parents[1]          # 01_code/
sys.path.insert(0, str(CODIGO))
from functions import validacion_clase as vc  # noqa: E402


def _coordenadas(n, semilla=0):
    rng = np.random.default_rng(semilla)
    return rng.uniform(0, 20_000, n), rng.uniform(0, 20_000, n)


class AleatoriaCV(unittest.TestCase):
    def test_cada_aviso_se_valida_una_vez(self):
        pliegues = vc.aleatoria_cv(103, v=5)
        self.assertEqual(len(pliegues), 5)
        validaciones = np.concatenate([va for _, va in pliegues])
        self.assertEqual(sorted(validaciones.tolist()), list(range(103)))

    def test_entrenamiento_y_validacion_son_disjuntos_y_completos(self):
        for tr, va in vc.aleatoria_cv(103, v=5):
            self.assertEqual(len(set(tr.tolist()) & set(va.tolist())), 0)
            self.assertEqual(len(tr) + len(va), 103)

    def test_misma_semilla_mismo_resultado(self):
        a = vc.aleatoria_cv(50)
        b = vc.aleatoria_cv(50)
        for (_, va_a), (_, va_b) in zip(a, b):
            np.testing.assert_array_equal(va_a, va_b)


class BloquesCV(unittest.TestCase):
    def test_validacion_cubre_cada_aviso_una_vez(self):
        x, y = _coordenadas(400)
        pliegues = vc.bloques_cv(x, y, v=5, tam=2000.0)
        validaciones = np.concatenate([va for _, va in pliegues])
        self.assertEqual(sorted(validaciones.tolist()), list(range(400)))

    def test_sin_buffer_entrenamiento_y_validacion_son_disjuntos(self):
        x, y = _coordenadas(400)
        for tr, va in vc.bloques_cv(x, y, v=5, tam=2000.0):
            self.assertEqual(len(set(tr.tolist()) & set(va.tolist())), 0)

    def test_buffer_retira_los_avisos_a_300_m_o_menos(self):
        x, y = _coordenadas(400)
        buffer = 300.0
        for tr, va in vc.bloques_cv(x, y, v=5, tam=2000.0, buffer=buffer):
            if len(va) == 0 or len(tr) == 0:
                continue
            dx = x[tr][:, None] - x[va][None, :]
            dy = y[tr][:, None] - y[va][None, :]
            distancia_minima = np.sqrt(dx ** 2 + dy ** 2).min(axis=1)
            self.assertTrue(np.all(distancia_minima > buffer))


class ClustersCV(unittest.TestCase):
    def test_diez_pliegues_cubren_cada_aviso_una_vez(self):
        x, y = _coordenadas(300)
        pliegues = vc.clusters_cv(x, y, v=10)
        self.assertEqual(len(pliegues), 10)
        validaciones = np.concatenate([va for _, va in pliegues])
        self.assertEqual(sorted(validaciones.tolist()), list(range(300)))

    def test_entrenamiento_excluye_el_cluster_de_validacion(self):
        x, y = _coordenadas(300)
        for tr, va in vc.clusters_cv(x, y, v=10):
            self.assertEqual(len(set(tr.tolist()) & set(va.tolist())), 0)


class LocalidadesCV(unittest.TestCase):
    def test_un_pliegue_por_localidad(self):
        grupo = np.array(["A", "A", "B", "C", "C", "C"])
        pliegues = vc.localidades_cv(grupo)
        self.assertEqual(len(pliegues), 3)
        for tr, va in pliegues:
            self.assertEqual(len(set(grupo[va].tolist())), 1)
            self.assertEqual(len(set(tr.tolist()) & set(va.tolist())), 0)


class ParticionInterna(unittest.TestCase):
    def setUp(self):
        self.x, self.y = _coordenadas(100)
        self.grupo = np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 20)

    def test_aleatoria_es_80_20(self):
        ajuste, parada = vc.particion_interna("aleatoria", self.x, self.y, self.grupo, semilla=2026)
        self.assertEqual(len(ajuste), 80)
        self.assertEqual(len(parada), 20)
        self.assertEqual(len(set(ajuste.tolist()) & set(parada.tolist())), 0)

    def test_bloques_ajuste_y_parada_son_disjuntos(self):
        ajuste, parada = vc.particion_interna("bloques", self.x, self.y, self.grupo, semilla=2026)
        self.assertGreater(len(ajuste), 0)
        self.assertGreater(len(parada), 0)
        self.assertEqual(len(set(ajuste.tolist()) & set(parada.tolist())), 0)

    def test_localidades_deja_fuera_una_sola(self):
        ajuste, parada = vc.particion_interna("localidades", self.x, self.y, self.grupo, semilla=2026)
        self.assertEqual(len(set(self.grupo[parada].tolist())), 1)
        self.assertEqual(len(set(self.grupo[ajuste].tolist()) & set(self.grupo[parada].tolist())), 0)

    def test_misma_semilla_mismo_resultado(self):
        a_ajuste, a_parada = vc.particion_interna("aleatoria", self.x, self.y, self.grupo, semilla=7)
        b_ajuste, b_parada = vc.particion_interna("aleatoria", self.x, self.y, self.grupo, semilla=7)
        np.testing.assert_array_equal(a_parada, b_parada)
        np.testing.assert_array_equal(a_ajuste, b_ajuste)

    def test_esquema_desconocido_falla(self):
        with self.assertRaises(ValueError):
            vc.particion_interna("no_existe", self.x, self.y, self.grupo, semilla=2026)


if __name__ == "__main__":
    unittest.main()
