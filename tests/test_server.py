"""Regression tests for sports-math OO-EPC 1.00 floor odds and fallback handling."""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

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


class TestConformalEngine(unittest.TestCase):
    @staticmethod
    def _make_records(n: int, start_date: str = "2026-10-01", outcome_fn=None, probs_fn=None):
        base = datetime(2026, 10, 1, 18, 0, tzinfo=timezone(timedelta(hours=8)))
        recs = []
        for i in range(n):
            dt = base + timedelta(days=i // 2, hours=(i % 2) * 3)
            probs = probs_fn(i) if probs_fn else [0.64, 0.21, 0.15]
            outcome = outcome_fn(i) if outcome_fn else ("H" if i % 5 != 0 else "D")
            recs.append({
                "match_id": str(3000000 + i),
                "sales_date": dt.strftime("%Y-%m-%d"),
                "kickoff_at": dt.isoformat(),
                "cutoff_time": (dt - timedelta(hours=3)).isoformat(),
                "league": "测试甲",
                "probs": probs,
                "outcome": outcome,
                "source_card": f"data/{dt.strftime('%Y-%m-%d')}/cards/{3000000 + i}.yaml",
            })
        return recs

    def test_happy_path_sets_and_venn_abers_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "ledger.jsonl"
            # 70% home hits (score 0.25) + 30% draw upsets (score 0.72): even if ACI rises to 0.28, q_hat=0.72
            recs = self._make_records(
                60,
                probs_fn=lambda i: [0.75, 0.15, 0.10] if (i % 10) < 7 else [0.52, 0.28, 0.20],
                outcome_fn=lambda i: "H" if (i % 10) < 7 else "D",
            )
            rec_res = _MOD.conformal_record_outcomes(recs, ledger_path=str(ledger))
            self.assertEqual(rec_res["status"], "ok")
            self.assertEqual(rec_res["added_count"], 60)

            cutoff = "2026-12-01T12:00:00+08:00"
            hi = _MOD.conformal_predict_1x2([0.74, 0.16, 0.10], cutoff_time=cutoff, ledger_path=str(ledger))
            self.assertEqual(hi["status"], "ok")
            self.assertEqual(hi["n_cal"], 60)
            self.assertEqual(hi["set_size"], 1)
            self.assertEqual(hi["tier"], "HIGH_CONVICTION")
            self.assertEqual(hi["venn_abers"]["status"], "ok")
            for c in ("H", "D", "A"):
                lo, up = hi["venn_abers"]["intervals_1x2"][c]
                self.assertTrue(0.0 < lo <= up < 1.0)

            chaos = _MOD.conformal_predict_1x2([0.35, 0.34, 0.31], cutoff_time=cutoff, ledger_path=str(ledger))
            self.assertEqual(chaos["set_size"], 3)
            self.assertEqual(chaos["tier"], "HIGH_ENTROPY_NO_PARLAY")

    def test_strict_causal_cutoff_blocks_future_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "ledger.jsonl"
            recs = self._make_records(25, outcome_fn=lambda i: "H")
            # Add future / same-day <3h upset records with huge LAC score 0.96
            recs.append({
                "match_id": "3999001",
                "sales_date": "2026-10-20",
                "kickoff_at": "2026-10-20T20:00:00+08:00",
                "league": "测试甲",
                "probs": [0.90, 0.06, 0.04],
                "outcome": "A",
                "source_card": "data/2026-10-20/cards/3999001.yaml",
            })
            _MOD.conformal_record_outcomes(recs, ledger_path=str(ledger))
            # Cutoff 18:00 is only 2h before 20:00 kickoff -> 3999001 must be excluded!
            res = _MOD.conformal_predict_1x2(
                [0.64, 0.21, 0.15],
                cutoff_time="2026-10-20T18:00:00+08:00",
                ledger_path=str(ledger),
            )
            self.assertEqual(res["n_cal"], 25)
            self.assertLess(res["q_hat_lac"], 0.50)

    def test_rejects_relative_ledger_path_without_env(self) -> None:
        old_env = _MOD.os.environ.pop("SPORTS_MATH_CONFORMAL_LEDGER", None)
        try:
            res = _MOD.conformal_predict_1x2(
                [0.60, 0.25, 0.15],
                cutoff_time="2026-11-26T12:00:00+08:00",
                ledger_path="data/conformal_calibration.jsonl",
            )
            self.assertEqual(res["status"], "error")
            self.assertEqual(res["error"]["error_code"], "INVALID_LEDGER_PATH")
        finally:
            if old_env is not None:
                _MOD.os.environ["SPORTS_MATH_CONFORMAL_LEDGER"] = old_env

    def test_small_sample_returns_insufficient_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "ledger.jsonl"
            _MOD.conformal_record_outcomes(self._make_records(19), ledger_path=str(ledger))
            res = _MOD.conformal_predict_1x2(
                [0.60, 0.25, 0.15],
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
            )
            self.assertEqual(res["status"], "insufficient_calibration")
            self.assertEqual(res["n_cal"], 19)
            self.assertIsNone(res["set_size"])
            self.assertIsNone(res["q_hat_lac"])
            lines = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertTrue(all(r["covered_lac"] is None for r in lines))

    def test_mondrian_draw_under_coverage_alert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "ledger.jsonl"
            # 15 early draws (high LAC 0.84) followed by 85 home wins (low LAC 0.20) -> q_hat=0.20 misses all draws
            recs = self._make_records(
                100,
                probs_fn=lambda i: [0.80, 0.16, 0.04],
                outcome_fn=lambda i: "D" if i < 15 else "H",
            )
            _MOD.conformal_record_outcomes(recs, ledger_path=str(ledger))
            res = _MOD.conformal_predict_1x2(
                [0.60, 0.25, 0.15],
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
            )
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["mondrian"]["class_counts"]["D"], 15)
            self.assertTrue(res["mondrian"]["draw_under_coverage_alert"])

    def test_idempotent_recording_replay_guard_and_aci_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "ledger.jsonl"
            recs = self._make_records(20, outcome_fn=lambda i: "H")
            r1 = _MOD.conformal_record_outcomes(recs, ledger_path=str(ledger))
            self.assertEqual(r1["added_count"], 20)
            # Duplicate call + known_answer_replay item
            dup = _MOD.conformal_record_outcomes(
                [recs[0], {**recs[0], "match_id": "3999999", "known_answer_replay": True}],
                ledger_path=str(ledger),
            )
            self.assertEqual(dup["added_count"], 0)
            self.assertEqual(len(dup["skipped"]), 2)

            # Feed 6 straight misses with frozen prediction_set=["H"] out-of-order -> alpha_t hits floor 0.10 cleanly
            misses = self._make_records(6, outcome_fn=lambda i: "A")
            for idx, m in enumerate(misses):
                dt = datetime(2026, 11, 20 + idx, 20, 0, tzinfo=timezone(timedelta(hours=8)))
                m["match_id"] = str(3500000 + idx)
                m["sales_date"] = dt.strftime("%Y-%m-%d")
                m["kickoff_at"] = dt.isoformat()
                m["cutoff_time"] = (dt - timedelta(hours=3)).isoformat()
                m["probs"] = [0.85, 0.10, 0.05]
                m["prediction_set"] = ["H"]
            res_miss = _MOD.conformal_record_outcomes(list(reversed(misses)), ledger_path=str(ledger))
            self.assertEqual(res_miss["added_count"], 6)
            self.assertEqual(res_miss["latest_alpha_t"], 0.10)
            ok, vmsg = _MOD.ConformalEngine.verify_ledger_file(str(ledger), min_rows=26)
            self.assertTrue(ok, vmsg)

    def test_p2_conformal_hardening_guards_and_alpha_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "ledger.jsonl"
            recs = self._make_records(25, outcome_fn=lambda i: "A")
            r0 = _MOD.conformal_record_outcomes(recs, ledger_path=str(ledger))
            self.assertEqual(r0["added_count"], 25)

            # Reject truthy-string replay, missing/malformed/non-prematch cutoff, bool/nan alpha_t
            base_rec = dict(recs[0])
            bad_cases = [
                {**base_rec, "match_id": "3991001", "known_answer_replay": "true"},
                {**base_rec, "match_id": "3991002", "cutoff_time": ""},
                {**base_rec, "match_id": "3991003", "cutoff_time": "not-a-date"},
                {**base_rec, "match_id": "3991004", "cutoff_time": base_rec["kickoff_at"]},
                {**base_rec, "match_id": "3991005", "alpha_t": True},
                {**base_rec, "match_id": "3991006", "alpha_t": float("nan")},
            ]
            bad_res = _MOD.conformal_record_outcomes(bad_cases, ledger_path=str(ledger))
            self.assertEqual(bad_res["added_count"], 0)
            self.assertEqual(len(bad_res["skipped"]), 6)
            reasons = {x["match_id"]: x["reason"] for x in bad_res["skipped"]}
            self.assertEqual(reasons["3991001"], "KNOWN_ANSWER_REPLAY_REJECTED")
            self.assertEqual(reasons["3991002"], "MISSING_OR_INVALID_CUTOFF")
            self.assertEqual(reasons["3991003"], "MISSING_OR_INVALID_CUTOFF")
            self.assertEqual(reasons["3991004"], "NON_PREMATCH_CUTOFF")
            self.assertEqual(reasons["3991005"], "INVALID_ALPHA_T")
            self.assertEqual(reasons["3991006"], "INVALID_ALPHA_T")

            # Stored ledger rows have alpha_target=0.20 and alpha_next=0.10 (5 straight misses),
            # so calling predict_1x2 with non-default alpha=0.12 must NOT silently adopt alpha_next=0.10.
            custom = _MOD.conformal_predict_1x2(
                [0.60, 0.25, 0.15],
                cutoff_time="2026-12-01T12:00:00+08:00",
                alpha=0.12,
                ledger_path=str(ledger),
            )
            self.assertEqual(custom["status"], "ok")
            self.assertAlmostEqual(custom["alpha_target"], 0.12, places=6)
            self.assertAlmostEqual(custom["alpha_t"], 0.12, places=6)


class TestBayesianTrackerEngine(unittest.TestCase):
    @staticmethod
    def _make_obs(
        entity_id: str,
        outcomes: list[int],
        p_baseline: float = 0.50,
        start_id: int = 3800000,
        entity_type: str | None = None,
        start_day: int = 1,
    ) -> list[dict]:
        base = datetime(2026, 10, start_day, 20, 0, tzinfo=timezone(timedelta(hours=8)))
        rows = []
        for idx, hit in enumerate(outcomes):
            dt = base + timedelta(days=idx)
            rec = {
                "match_id": str(start_id + idx),
                "sales_date": dt.strftime("%Y-%m-%d"),
                "kickoff_at": dt.isoformat(),
                "cutoff_time": (dt - timedelta(hours=3)).isoformat(),
                "entity_id": entity_id,
                "direction": "主胜",
                "p_baseline": p_baseline,
                "hit": hit,
                "source_card": f"data/{dt.strftime('%Y-%m-%d')}/cards/{start_id + idx}.yaml",
            }
            if entity_type is not None:
                rec["entity_type"] = entity_type
            rows.append(rec)
        return rows

    def test_beta_math_closed_forms(self) -> None:
        self.assertAlmostEqual(_MOD.regularized_incomplete_beta(0.3, 1.0, 5.0), 1.0 - (0.7 ** 5), places=9)
        self.assertAlmostEqual(_MOD.regularized_incomplete_beta(0.4, 4.0, 1.0), 0.4 ** 4, places=9)
        self.assertAlmostEqual(_MOD.regularized_incomplete_beta(0.5, 7.5, 7.5), 0.5, places=9)
        self.assertAlmostEqual(_MOD.beta_ppf(0.5, 7.5, 7.5), 0.5, places=9)

    def test_ae1_heavy_favorite_mirage_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            obs = self._make_obs("H9", [1, 1, 1, 1, 1], p_baseline=0.75)
            rec_res = _MOD.bayesian_record_outcomes(obs, ledger_path=str(ledger))
            self.assertEqual(rec_res["status"], "ok")
            self.assertEqual(rec_res["added_count"], 5)
            rep = _MOD.bayesian_health_report(ledger_path=str(ledger), entity_ids=["H9"])
            self.assertEqual(rep["status"], "ok")
            ent = rep["entities"][0]
            self.assertLess(ent["e_pos"], 2.0)
            self.assertEqual(ent["advisory_status"], "COLLECTING")

    def test_ae2_underdog_true_alpha_and_strong_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            obs = self._make_obs("H8", [1, 1, 0, 1, 1], p_baseline=0.30)
            _MOD.bayesian_record_outcomes(obs, ledger_path=str(ledger))
            rep = _MOD.bayesian_health_report(ledger_path=str(ledger), entity_ids=["H8"])
            ent = rep["entities"][0]
            self.assertGreaterEqual(ent["p_beat_market"], 0.90)
            self.assertGreaterEqual(ent["e_pos"], 5.0)
            self.assertEqual(ent["advisory_status"], "PROMOTE_RECOMMENDED")

            more = self._make_obs("H8", [1, 1], p_baseline=0.30, start_id=3800100, start_day=10)
            _MOD.bayesian_record_outcomes(more, ledger_path=str(ledger))
            rep2 = _MOD.bayesian_health_report(ledger_path=str(ledger), entity_ids=["H8"])
            self.assertGreaterEqual(rep2["entities"][0]["e_pos"], 20.0)
            self.assertEqual(rep2["entities"][0]["advisory_status"], "STRONG_EVIDENCE")

    def test_draft_deprecation_and_zero_obs_stale_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            obs = self._make_obs("H7", [0, 0, 0, 0, 0, 0], p_baseline=0.55)
            _MOD.bayesian_record_outcomes(obs, ledger_path=str(ledger))
            rep = _MOD.bayesian_health_report(
                ledger_path=str(ledger),
                entity_metadata={
                    "H1": {"entity_type": "draft", "stale_after": "2026-11-26T00:00:00Z"},
                    "H2": {"entity_type": "draft", "stale_after": "2026-12-10T00:00:00Z"},
                },
                as_of="2026-11-26T10:00:00+08:00",
            )
            by_id = {e["entity_id"]: e for e in rep["entities"]}
            self.assertGreaterEqual(by_id["H7"]["e_neg"], 5.0)
            self.assertGreaterEqual(by_id["H7"]["p_below_market"], 0.85)
            self.assertEqual(by_id["H7"]["advisory_status"], "DEPRECATE_RECOMMENDED")
            self.assertEqual(by_id["H1"]["n_obs"], 0)
            self.assertEqual(by_id["H1"]["advisory_status"], "STALE_REVIEW")
            self.assertEqual(by_id["H2"]["advisory_status"], "COLLECTING")

    def test_ae3_regime_drift_cusum_and_pure_decayed_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            # 20 wins followed by 10 losses -> cusum_drift_e catches drift despite win cushion
            obs_cushion = self._make_obs(
                "playbooks/false-favorite-trap.md",
                ([1] * 20) + ([0] * 10),
                p_baseline=0.50,
            )
            # 15 consecutive misses -> decayed_p_below_market >= 0.90 branch directly
            obs_pure = self._make_obs(
                "playbooks/head-fake.md",
                [0] * 15,
                p_baseline=0.50,
                start_id=3900000,
            )
            _MOD.bayesian_record_outcomes(obs_cushion + obs_pure, ledger_path=str(ledger))
            rep = _MOD.bayesian_health_report(ledger_path=str(ledger))
            by_id = {e["entity_id"]: e for e in rep["entities"]}

            pb1 = by_id["playbooks/false-favorite-trap.md"]
            self.assertGreaterEqual(pb1["cusum_drift_e"], 8.0)
            self.assertGreater(pb1["decayed_p_below_market"], 0.50)
            self.assertEqual(pb1["advisory_status"], "DRIFT_ALERT")

            pb2 = by_id["playbooks/head-fake.md"]
            self.assertGreaterEqual(pb2["decayed_p_below_market"], 0.90)
            self.assertEqual(pb2["advisory_status"], "DRIFT_ALERT")

    def test_entity_normalization_idempotency_clamping_and_ae4_guards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            obs = [
                {
                    "match_id": "3700001",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T20:00:00+08:00",
                    "cutoff_time": "2026-11-20T17:00:00+08:00",
                    "entity_id": "wiki/playbooks/07-false-favorite-trap.md#not",
                    "direction": "主胜",
                    "p_baseline": 0.995,  # clamped to 0.98
                    "hit": 1,
                    "source_card": "data/2026-11-20/cards/3700001.yaml",
                },
                {
                    "match_id": "3700001",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T20:00:00+08:00",
                    "cutoff_time": "2026-11-20T17:00:00+08:00",
                    "entity_id": "skills/deep-analysis/wiki/playbooks/false-favorite-trap.md",
                    "direction": "主胜",
                    "p_baseline": 0.80,
                    "hit": 1,
                    "source_card": "data/2026-11-20/cards/3700001.yaml",
                },
                {
                    "match_id": "3700002",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T22:00:00+08:00",
                    "cutoff_time": "2026-11-20T18:00:00+08:00",
                    "entity_id": "features/odds-timeseries-dynamics.md",
                    "direction": "客胜",
                    "p_baseline": 0.005,  # clamped to 0.02
                    "hit": 0,
                    "source_card": "data/2026-11-20/cards/3700002.yaml",
                },
                {
                    "match_id": "3700003",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T22:00:00+08:00",
                    "cutoff_time": "2026-11-20T18:00:00+08:00",
                    "entity_id": "portfolio:1x2_primary",
                    "direction": "主胜",
                    "p_baseline": 0.62,
                    "hit": 1,
                    "source_card": "data/2026-11-20/cards/3700003.yaml",
                },
                # Invalid AE4 cases
                {
                    "match_id": "3700004",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T22:00:00+08:00",
                    "cutoff_time": "2026-11-20T18:00:00+08:00",
                    "entity_id": "H1",
                    "direction": "主胜",
                    "p_baseline": 0.45,
                    "hit": 1,
                    "known_answer_replay": "true",
                },
                {
                    "match_id": "3700005",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T22:00:00+08:00",
                    "cutoff_time": "2026-11-20T22:00:00+08:00",
                    "entity_id": "H1",
                    "direction": "主胜",
                    "p_baseline": 0.45,
                    "hit": 1,
                },
                {
                    "match_id": "3700006",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T22:00:00+08:00",
                    "cutoff_time": "",
                    "entity_id": "H1",
                    "direction": "主胜",
                    "p_baseline": 0.45,
                    "hit": 1,
                },
                {
                    "match_id": "3700007",
                    "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T22:00:00+08:00",
                    "cutoff_time": "2026-11-20T18:00:00+08:00",
                    "entity_id": "H1",
                    "direction": "主胜",
                    "p_baseline": 1.0,
                    "hit": 1,
                },
            ]
            res = _MOD.bayesian_record_outcomes(obs, ledger_path=str(ledger))
            self.assertEqual(res["added_count"], 3)
            self.assertEqual(len(res["skipped"]), 5)
            rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(rows[0]["entity_id"], "playbooks/false-favorite-trap.md")
            self.assertEqual(rows[0]["p_baseline"], 0.98)
            self.assertEqual(rows[1]["entity_id"], "features/odds-timeseries-dynamics.md")
            self.assertEqual(rows[1]["p_baseline"], 0.02)
            self.assertEqual(rows[2]["entity_id"], "portfolio:1x2_primary")
            ok, vmsg = _MOD.BayesianTrackerEngine.verify_ledger_file(str(ledger), min_rows=3)
            self.assertTrue(ok, vmsg)

    def test_portfolio_drift_watch_and_alert_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            # 3 losses at p0=0.65 -> cusum ≈ 3.98 < 5.0 -> stays HEALTHY (no false DRIFT_WATCH spam)
            _MOD.bayesian_record_outcomes(
                self._make_obs("portfolio:1x2_primary", [0, 0, 0], p_baseline=0.65),
                ledger_path=str(ledger),
            )
            rep_healthy = _MOD.bayesian_health_report(ledger_path=str(ledger), entity_type="portfolio")
            self.assertEqual(rep_healthy["entities"][0]["advisory_status"], "HEALTHY")

            # Add 5 more consecutive losses -> 8 obs (decayed_n_eff≈6.73 < 8.0) -> DRIFT_WATCH
            _MOD.bayesian_record_outcomes(
                self._make_obs("portfolio:1x2_primary", [0, 0, 0, 0, 0], p_baseline=0.65, start_id=3800100, start_day=10),
                ledger_path=str(ledger),
            )
            rep_watch = _MOD.bayesian_health_report(ledger_path=str(ledger), entity_type="portfolio")
            self.assertEqual(rep_watch["entities"][0]["advisory_status"], "DRIFT_WATCH")

            # Add 2 more consecutive losses -> 10 obs (decayed_n_eff≈8.025 >= 8.0, cusum >= 8.0) -> DRIFT_ALERT
            _MOD.bayesian_record_outcomes(
                self._make_obs("portfolio:1x2_primary", [0, 0], p_baseline=0.65, start_id=3800200, start_day=20),
                ledger_path=str(ledger),
            )
            rep_alert = _MOD.bayesian_health_report(ledger_path=str(ledger), entity_type="portfolio")
            self.assertEqual(rep_alert["entities"][0]["advisory_status"], "DRIFT_ALERT")

    def test_rejects_relative_path_and_empty_records(self) -> None:
        old_env = _MOD.os.environ.pop("SPORTS_MATH_BAYESIAN_LEDGER", None)
        try:
            res_empty = _MOD.bayesian_record_outcomes([], ledger_path="data/bayesian_tracker.jsonl")
            self.assertEqual(res_empty["status"], "error")
            self.assertEqual(res_empty["error"]["error_code"], "EMPTY_RECORDS")

            res_rel = _MOD.bayesian_health_report(ledger_path="data/bayesian_tracker.jsonl")
            self.assertEqual(res_rel["status"], "error")
            self.assertEqual(res_rel["error"]["error_code"], "INVALID_LEDGER_PATH")
        finally:
            if old_env is not None:
                _MOD.os.environ["SPORTS_MATH_BAYESIAN_LEDGER"] = old_env

    def test_stdio_mcp_subprocess_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c_ledger = pathlib.Path(td) / "conformal.jsonl"
            b_ledger = pathlib.Path(td) / "bayesian.jsonl"
            _MOD.conformal_record_outcomes(TestConformalEngine._make_records(25), ledger_path=str(c_ledger))
            proc = subprocess.Popen(
                [sys.executable, str(_SERVER_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            try:
                assert proc.stdin and proc.stdout

                def _rpc(msg: dict, expect_id: int | None = None) -> dict | None:
                    assert proc.stdin and proc.stdout
                    proc.stdin.write(json.dumps(msg) + "\n")
                    proc.stdin.flush()
                    if expect_id is None:
                        return None
                    for _ in range(15):
                        line = proc.stdout.readline().strip()
                        if not line:
                            break
                        obj = json.loads(line)
                        if obj.get("id") == expect_id:
                            return obj
                    self.fail(f"Missing RPC response id={expect_id}")

                init_resp = _rpc(
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}},
                    expect_id=1,
                )
                self.assertIsNotNone(init_resp)
                _rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

                r2 = _rpc(
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "conformal_predict_1x2", "arguments": {"probs": [0.70, 0.18, 0.12], "cutoff_time": "2026-12-01T12:00:00+08:00", "ledger_path": str(c_ledger)}}},
                    expect_id=2,
                )
                c_payload = json.loads(r2["result"]["content"][0]["text"])
                self.assertEqual(c_payload["status"], "ok")
                self.assertEqual(c_payload["set_size"], 1)

                r3 = _rpc(
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "bayesian_record_outcomes", "arguments": {"records": self._make_obs("H8", [1, 1, 0, 1, 1], p_baseline=0.30), "ledger_path": str(b_ledger)}}},
                    expect_id=3,
                )
                br_payload = json.loads(r3["result"]["content"][0]["text"])
                self.assertEqual(br_payload["status"], "ok")
                self.assertEqual(br_payload["added_count"], 5)

                r4 = _rpc(
                    {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "bayesian_health_report", "arguments": {"ledger_path": str(b_ledger), "entity_ids": ["H8"]}}},
                    expect_id=4,
                )
                bh_payload = json.loads(r4["result"]["content"][0]["text"])
                self.assertEqual(bh_payload["status"], "ok")
                ent = bh_payload["entities"][0]
                expected_keys = {
                    "entity_id", "entity_type", "n_obs", "hits", "mean_p_baseline",
                    "posterior_mean", "ci90", "p_beat_market", "p_below_market",
                    "e_pos", "e_neg", "anytime_p_pos", "decayed_n_eff",
                    "decayed_posterior_mean", "decayed_p_below_market",
                    "cusum_drift_e", "advisory_status",
                }
                self.assertTrue(expected_keys.issubset(ent.keys()))
                self.assertEqual(ent["advisory_status"], "PROMOTE_RECOMMENDED")
            finally:
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
                proc.terminate()
                proc.wait(timeout=5)


class TestProbsDictCompatibility(unittest.TestCase):
    def test_normalize_probs_accepts_hda_and_home_draw_away_dicts(self) -> None:
        p1, err1 = _MOD.ConformalEngine.normalize_probs({"H": 0.50, "D": 0.30, "A": 0.20})
        self.assertIsNone(err1)
        self.assertEqual([round(x, 6) for x in p1], [0.5, 0.3, 0.2])

        p2, err2 = _MOD.ConformalEngine.normalize_probs({"home": 19.18, "draw": 20.71, "away": 60.11})
        self.assertIsNone(err2)
        self.assertAlmostEqual(p2[0], 0.1918, places=4)
        self.assertAlmostEqual(p2[1], 0.2071, places=4)
        self.assertAlmostEqual(p2[2], 0.6011, places=4)

        p3, err3 = _MOD.ConformalEngine.normalize_probs({"h": 0.4, "d": 0.3, "a": 0.3})
        self.assertIsNone(err3)
        self.assertEqual([round(x, 6) for x in p3], [0.4, 0.3, 0.3])

    def test_normalize_probs_rejects_malformed_dicts(self) -> None:
        for bad in (
            {"H": 0.5, "D": 0.5},
            {"H": 0.5, "D": 0.3, "A": 0.2, "extra": 0.0},
            {"H": 0.5, "draw": 0.3, "A": 0.2},
            {},
        ):
            val, err = _MOD.ConformalEngine.normalize_probs(bad, context="bad_probs")
            self.assertIsNone(val)
            self.assertIsNotNone(err)
            self.assertEqual(err["error_code"], "INVALID_PROBS_DICT")

    def test_record_outcomes_accepts_dict_probs_and_persists_canonical_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "conformal.jsonl"
            recs = [
                {
                    "match_id": "3990001",
                    "sales_date": "2026-11-26",
                    "kickoff_at": "2026-11-27T01:45:00+08:00",
                    "cutoff_time": "2026-11-26T20:39:22+08:00",
                    "league": "欧罗巴杯",
                    "probs": {"H": 0.1918, "D": 0.2071, "A": 0.6011},
                    "outcome": "A",
                    "source_card": "data/2026-11-26/cards/3990001.yaml",
                    "known_answer_replay": False,
                },
                {
                    "match_id": "3990002",
                    "sales_date": "2026-11-26",
                    "kickoff_at": "2026-11-27T04:00:00+08:00",
                    "cutoff_time": "2026-11-26T20:39:22+08:00",
                    "league": "欧罗巴杯",
                    "probs": {"home": 0.7445, "draw": 0.1487, "away": 0.1068},
                    "outcome": "H",
                    "source_card": "data/2026-11-26/cards/3990002.yaml",
                    "known_answer_replay": False,
                },
            ]
            res = _MOD.conformal_record_outcomes(recs, ledger_path=str(ledger))
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["added_count"], 2)
            self.assertEqual(res["skipped"], [])
            rows, rerr = _MOD.ConformalEngine.load_ledger(ledger)
            self.assertIsNone(rerr)
            self.assertEqual(len(rows), 2)
            self.assertIsInstance(rows[0]["probs"], list)
            self.assertEqual(rows[0]["probs"], [0.1918, 0.2071, 0.6011])
            self.assertEqual(rows[1]["probs"], [0.7445, 0.1487, 0.1068])

    def test_predict_1x2_accepts_dict_probs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "conformal.jsonl"
            _MOD.conformal_record_outcomes(TestConformalEngine._make_records(25), ledger_path=str(ledger))
            res = _MOD.conformal_predict_1x2(
                {"H": 0.74, "D": 0.16, "A": 0.10},
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
            )
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["set_size"], 1)
            self.assertEqual(res["prediction_set"], ["H"])


class TestSyncDayLedgers(unittest.TestCase):
    @staticmethod
    def _build_fixture_repo(root: pathlib.Path) -> None:
        day = root / "data" / "2026-11-26"
        cards = day / "cards"
        cards.mkdir(parents=True)
        (root / "skills" / "deep-analysis" / "wiki").mkdir(parents=True)
        (root / "skills" / "deep-analysis" / "wiki" / "index.md").write_text(
            "## 四、待验证池（OKF draft）\n"
            "* [H1 · 测试假设1](playbooks/head-fake.md) - `Gap-1｜stale_after: 2026-11-26T00:00:00Z｜draft`\n"
            "* [H5 · 测试假设5](playbooks/head-fake.md) - `Gap-3｜stale_after: 2026-12-17T00:00:00Z｜draft`\n",
            encoding="utf-8",
        )
        (day / "meta.md").write_text(
            "<!-- SCRAPER:SCHEDULE:START -->\n"
            "| 场次编号 | 比赛 ID | 联赛 | 主队 | 客队 | 开球时间 | 状态 | 数据状态 | Polymarket |\n"
            "| 周四004 | 3073720 | 欧罗巴杯 | 比尔森胜利 | 本菲卡 | 2026-11-27 01:45 | 已完场 (4:0, 半3:0) | [已推演] | [link](https://polymarket.com/event/x) |\n"
            "| 周四005 | 3073761 | 欧罗巴杯 | 塞尔塔 | 伯恩茅斯 | 2026-11-27 01:45 | 已完场 (1:2, 半1:0) | [已推演] | [link](https://polymarket.com/event/y) |\n"
            "<!-- SCRAPER:SCHEDULE:END -->\n",
            encoding="utf-8",
        )
        (cards / "3073720.yaml").write_text(
            "worker_receipt: { state: 'completed', matchId: '3073720' }\n"
            "announce:\n"
            "  memory_summary:\n"
            "    identity: { matchId: '3073720', league: '欧罗巴杯', homeTeam: '比尔森胜利', awayTeam: '本菲卡', kickoffTime: '2026-11-27 01:45' }\n"
            "    regime: { league_tier: '洲际杯赛', ah_tier_and_refund: '深盘(-1.25) ; half_refund' }\n"
            "    data_ref: { odds_cutoff: '2026-11-26T20:39:22+08:00', research: { analysis_phase: 'prematch' } }\n"
            "    decision:\n"
            "      primary: '负'\n"
            "      secondary: '平'\n"
            "      probabilities: { home: 0.1557, draw: 0.1809, away: 0.6634 }\n"
            "      conformal: { prediction_set: ['A'], alpha_t: 0.12 }\n"
            "    wiki_hits:\n"
            "      - { role: 'residual', page: 'skills/deep-analysis/wiki/playbooks/deep-water-resistance.md', direction: '平(纳入次选对冲)' }\n"
            "      - { role: 'feature', page: 'skills/deep-analysis/wiki/features/asian-handicap-euro-mapping.md', direction: '受让不败' }\n",
            encoding="utf-8",
        )
        (cards / "3073761.yaml").write_text(
            "worker_receipt: { state: 'completed', matchId: '3073761' }\n"
            "announce:\n"
            "  memory_summary:\n"
            "    identity: { matchId: '3073761', league: '欧罗巴杯', homeTeam: '塞尔塔', awayTeam: '伯恩茅斯', kickoffTime: '2026-11-27 01:45' }\n"
            "    regime: { league_tier: '洲际杯赛', ah_tier_and_refund: '中盘 ; no_refund' }\n"
            "    data_ref: { odds_cutoff: '2026-11-26T20:39:22+08:00', research: { analysis_phase: 'prematch' } }\n"
            "    decision:\n"
            "      primary: '负'\n"
            "      secondary: '平'\n"
            "      three_dimensions: { margin: '伯恩茅斯 -0.5' }\n"
            "      probabilities: { home: 0.2354, draw: 0.2561, away: 0.5084 }\n"
            "      conformal: { prediction_set: ['H', 'D', 'A'], alpha_t: 0.12 }\n"
            "    wiki_hits:\n"
            "      - { role: 'primary', page: 'skills/deep-analysis/wiki/playbooks/head-fake.md', direction: '客胜' }\n",
            encoding="utf-8",
        )

    def test_sync_day_ledgers_idempotent_compact_and_auto_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            self._build_fixture_repo(root)
            r1 = _MOD.sync_day_ledgers("2026-11-26", str(root), as_of="2026-11-27T04:00:00Z")
            self.assertEqual(r1["status"], "ok")
            self.assertEqual(r1["settled_card_count"], 2)
            self.assertEqual(r1["conformal"]["added_count"], 2)
            self.assertEqual(r1["bayesian"]["added_count"], 4)  # 2 portfolio + deep-water (parenthetical stripped!) + head-fake
            self.assertIn("H1", r1["health"]["alerts"]["stale_reviews"])
            self.assertIn("H5", r1["health"]["zero_obs_drafts"].get("COLLECTING", []))
            self.assertLess(len(json.dumps(r1, ensure_ascii=False).encode("utf-8")), 4000)

            # Second call is 100% idempotent
            r2 = _MOD.sync_day_ledgers("2026-11-26", str(root), as_of="2026-11-27T04:00:00Z")
            self.assertEqual(r2["status"], "ok")
            self.assertEqual(r2["conformal"]["added_count"], 0)
            self.assertEqual(r2["conformal"]["skipped_count"], 2)
            self.assertEqual(r2["bayesian"]["added_count"], 0)
            self.assertEqual(r2["bayesian"]["skipped_count"], 4)

    def test_sync_day_ledgers_structured_errors(self) -> None:
        bad_date = _MOD.sync_day_ledgers("2026/11/26", "C:/nonexistent")
        self.assertEqual(bad_date["status"], "error")
        self.assertEqual(bad_date["error"]["error_code"], "INVALID_SALES_DATE")
        bad_root = _MOD.sync_day_ledgers("2026-11-26", "relative/path")
        self.assertEqual(bad_root["status"], "error")
        self.assertEqual(bad_root["error"]["error_code"], "INVALID_REPO_ROOT")

    def test_sync_day_ledgers_stdio_mcp_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            self._build_fixture_repo(root)
            proc = subprocess.Popen(
                [sys.executable, str(_SERVER_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            try:
                assert proc.stdin and proc.stdout

                def _rpc(msg: dict, expect_id: int | None = None) -> dict | None:
                    assert proc.stdin and proc.stdout
                    proc.stdin.write(json.dumps(msg) + "\n")
                    proc.stdin.flush()
                    if expect_id is None:
                        return None
                    for _ in range(15):
                        line = proc.stdout.readline().strip()
                        if not line:
                            break
                        obj = json.loads(line)
                        if obj.get("id") == expect_id:
                            return obj
                    self.fail(f"Missing RPC response id={expect_id}")

                _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}, expect_id=1)
                _rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
                resp = _rpc(
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "sync_day_ledgers", "arguments": {"sales_date": "2026-11-26", "repo_root": str(root), "as_of": "2026-11-27T04:00:00Z"}}},
                    expect_id=2,
                )
                payload = json.loads(resp["result"]["content"][0]["text"])
                self.assertEqual(payload["status"], "ok")
                self.assertEqual(payload["conformal"]["added_count"], 2)
                self.assertEqual(payload["bayesian"]["added_count"], 4)
            finally:
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
                proc.terminate()
                proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
