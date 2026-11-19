"""Regression tests for sports-math OO-EPC 1.00 floor odds and fallback handling."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest

_SERVER_PATH = pathlib.Path(__file__).resolve().parents[1] / "server.py"
_SPEC = importlib.util.spec_from_file_location("sports_math_server", _SERVER_PATH)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


class TestOoEpcFloorOdds(unittest.TestCase):
    def test_rejects_odds_below_one(self) -> None:
        res = _MOD.devig_oo_epc(0.99, 10.0, 20.0)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"]["error_code"], "ODD_OUT_OF_RANGE")

    def test_accepts_one_point_zero_floor_odds_hkjc(self) -> None:
        res = _MOD.devig_oo_epc(1.00, 17.5, 51.0)
        self.assertNotIn("error", res)
        probs = res["probabilities"]
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(all(p > 0.0 for p in probs))
        invs = [1.0, 1.0 / 17.5, 1.0 / 51.0]
        booksum = sum(invs)
        for p, inv in zip(probs, invs):
            self.assertAlmostEqual(p, inv / booksum, places=5)

    def test_accepts_symmetric_one_point_zero_without_float_leak(self) -> None:
        probs = _MOD.OoEpcCalculator.calculate([1.00, 3.00, 3.00])
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertTrue(all(p > 0.0 and math.isfinite(p) for p in probs))
        self.assertAlmostEqual(probs[0], 0.6, places=6)

    def test_batch_handles_hkjc_one_point_zero(self) -> None:
        res = _MOD.devig_oo_epc_batch(
            [
                {"cid": 177, "name": "平博", "open": [1.03, 18.0, 67.0], "latest": [1.02, 18.5, 67.0]},
                {"cid": 432, "name": "香港马会", "open": [1.00, 17.5, 51.0], "latest": [1.00, 17.5, 51.0]},
            ]
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["count"], 2)


if __name__ == "__main__":
    unittest.main()
