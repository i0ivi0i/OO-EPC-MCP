"""Tests for sports-math v1.4.0 modularization, adversarial hardening, and contextual Bayesian exploration."""

from __future__ import annotations

import importlib.util
import json
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


class TestContextualBayesianExploration(unittest.TestCase):
    @staticmethod
    def _make_ctx_obs(
        entity_id: str,
        hits: list[int],
        p_baseline: float = 0.50,
        start_id: int = 3900000,
        start_dt: datetime | None = None,
        league: str = "英超",
        league_tier: str = "五大联赛",
        ah_tier_and_refund: str = "中盘(-0.75) ; half_refund",
        context_bucket: str | None = None,
        observation_role: str = "primary",
        known_answer_replay: bool = False,
        cutoff_before_ko_hours: int = 3,
    ) -> list[dict]:
        base = start_dt or datetime(2026, 10, 1, 18, 0, tzinfo=timezone(timedelta(hours=8)))
        recs = []
        for i, hit in enumerate(hits):
            ko = base + timedelta(days=i)
            rec = {
                "match_id": str(start_id + i),
                "sales_date": ko.strftime("%Y-%m-%d"),
                "kickoff_at": ko.isoformat(),
                "cutoff_time": (ko - timedelta(hours=cutoff_before_ko_hours)).isoformat(),
                "entity_id": entity_id,
                "p_baseline": p_baseline,
                "hit": hit,
                "observation_role": observation_role,
                "known_answer_replay": known_answer_replay,
            }
            if context_bucket is not None:
                rec["context_bucket"] = context_bucket
            else:
                rec["league"] = league
                rec["league_tier"] = league_tier
                rec["ah_tier_and_refund"] = ah_tier_and_refund
            recs.append(rec)
        return recs

    def test_modular_exports_and_version(self) -> None:
        from engines import (
            BayesianTrackerEngine,
            ConformalEngine,
            OoEpcCalculator,
            PoissonEngine,
            evaluate_context,
            normalize_context_bucket,
            select_playbooks,
        )

        self.assertIs(_MOD.OoEpcCalculator, OoEpcCalculator)
        self.assertIs(_MOD.PoissonEngine, PoissonEngine)
        self.assertIs(_MOD.ConformalEngine, ConformalEngine)
        self.assertIs(_MOD.BayesianTrackerEngine, BayesianTrackerEngine)
        self.assertIs(_MOD.normalize_context_bucket, normalize_context_bucket)
        self.assertIs(_MOD.evaluate_context, evaluate_context)
        self.assertIs(_MOD.select_playbooks, select_playbooks)
        for sym in ("_is_truthy_flag", "_is_strict_finite_number", "_betacf", "regularized_incomplete_beta", "beta_ppf"):
            self.assertTrue(hasattr(_MOD, sym), f"Missing re-export {sym}")
        self.assertIn('version="1.5.0"', _SERVER_PATH.read_text(encoding="utf-8"))

    def test_normalize_context_bucket_multi_clause_and_adversarial_formats(self) -> None:
        norm = _MOD.normalize_context_bucket
        self.assertEqual(norm(league="英超", league_tier="五大联赛", ah_tier_and_refund="深盘(-1.5/2) ; half_refund"), "top5:deep")
        self.assertEqual(norm(league="欧罗巴", league_tier="洲际杯赛", ah_tier_and_refund="均势平手(0.0) ; full_refund"), "continental_cup:even_shallow")
        self.assertEqual(norm(league="欧国联", league_tier="国家队赛事", ah_tier_and_refund="中盘(-0.75) ; half_refund"), "national_team:mid")
        self.assertEqual(norm(league="荷乙", league_tier="次级联赛", ah_tier_and_refund="初盘 -0.5 -> 最新 0.0 (188Bet, 非原1.0)"), "other_domestic:even_shallow")
        self.assertEqual(norm(league="法甲", league_tier="五大联赛", ah_tier_and_refund="浅盘 -> 现主流中盘(-0.75) ; half_refund"), "top5:mid")
        self.assertEqual(norm(league="意甲", league_tier="五大联赛", ah_tier_and_refund="初盘深盘(-1.25) ; 终盘中盘(-1.0) ; full_refund"), "top5:mid")
        self.assertEqual(norm(league="日职", league_tier="", ah_tier_and_refund="0/0.5 (177/281) full_refund 非原1.5"), "other_domestic:even_shallow")
        self.assertEqual(norm(league="世预赛", league_tier="国家队赛事", ah_tier_and_refund="初主让0.75中深缓冲→主让0.25浅盘；half_refund；当前打平输半，非原0.75赢1球赢半"), "national_team:even_shallow")
        # Adversarial real-card cases caught by code review:
        self.assertEqual(norm(league="巴西甲", league_tier="次级联赛", ah_tier_and_refund="平手 ; full_refund"), "other_domestic:even_shallow")
        self.assertEqual(norm(league="巴西甲", league_tier="巴西顶级联赛（原四类枚举未覆盖，保留源联赛不误标次级）", ah_tier_and_refund="浅盘/主让平半 ; half_refund"), "other_domestic:even_shallow")
        self.assertEqual(norm(league="欧冠杯", league_tier="洲际杯赛", ah_tier_and_refund="深盘初盘(-1.5/-1.75)退至整球/四分盘(-1.0/-0.75) ; full_refund(6/7) | half_refund(42)"), "continental_cup:mid")
        self.assertEqual(norm(league="欧国联", league_tier="国家队赛事", ah_tier_and_refund="主让0.5，退档自0.5/1，半球无退本；让球结论观望"), "national_team:even_shallow")
        self.assertEqual(norm(league="欧国联", league_tier="国家队赛事", ah_tier_and_refund="半球浅盘/错盘；no_refund，马会半一为half_refund"), "national_team:even_shallow")
        self.assertEqual(norm(league="欧国联", league_tier="国家队赛事", ah_tier_and_refund="47/12初主让0.25 half_refund；1/3/8/42初主让0.5 no_refund；最新六家主让0.5 no_refund"), "national_team:even_shallow")
        self.assertEqual(norm(league="日联杯", league_tier="次级联赛", ah_tier_and_refund="浅盘/生死盘（主让-0.5主流6家，马会-0.25）; no_refund（马会48为half_refund）"), "other_domestic:even_shallow")
        self.assertEqual(norm(league="欧国联", league_tier="国家队赛事", ah_tier_and_refund="主让半球浅/中盘；no_refund；六家半球、马会半一"), "national_team:even_shallow")
        self.assertEqual(norm(league="西甲", league_tier="五大联赛", ah_tier_and_refund="中盘/客让半球 ; no_refund"), "top5:even_shallow")
        self.assertEqual(norm(league="", league_tier="", ah_tier_and_refund=""), "unknown")

    def test_parse_timestamp_fractional_seconds_and_invalid_calendar(self) -> None:
        from engines.common import clean_str, parse_timestamp

        dt = parse_timestamp("2026-11-26T04:00:00.123456Z")
        self.assertIsNotNone(dt)
        assert dt is not None
        self.assertEqual(dt, datetime(2026, 11, 26, 4, 0, 0, tzinfo=timezone.utc))
        self.assertIsNone(parse_timestamp("2026-02-30T12:00:00Z"))
        self.assertEqual(clean_str(None), "")

    def test_ae1_single_shrinkage_logit_transfer_and_unknown_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            rows_global = self._make_ctx_obs(
                "playbooks/false-favorite-trap.md",
                [1] * 12 + [0] * 3,
                p_baseline=0.50,
                start_id=3910000,
                league="英超",
                league_tier="五大联赛",
                ah_tier_and_refund="深盘(-1.5) ; full_refund",
            )
            res = _MOD.bayesian_record_outcomes(rows_global, ledger_path=str(ledger))
            self.assertEqual(res["status"], "ok")

            loaded, err = _MOD.BayesianTrackerEngine.load_ledger(ledger)
            self.assertIsNone(err)
            ev_zero = _MOD.evaluate_context(loaded, "playbooks/false-favorite-trap.md", "national_team:mid")
            self.assertAlmostEqual(ev_zero["p_0c_anchor"], 0.50, places=6)
            self.assertAlmostEqual(ev_zero["prior_center"], 0.7492, places=3)

            res2 = _MOD.bayesian_record_outcomes(
                self._make_ctx_obs(
                    "playbooks/false-favorite-trap.md",
                    [1],
                    p_baseline=0.40,
                    start_id=3910100,
                    league="欧罗巴",
                    league_tier="洲际杯赛",
                    ah_tier_and_refund="平手(0.0) ; full_refund",
                ),
                ledger_path=str(ledger),
            )
            self.assertEqual(res2["status"], "ok")
            loaded_ctx, _ = _MOD.BayesianTrackerEngine.load_ledger(ledger)
            ev_ctx = _MOD.evaluate_context(loaded_ctx, "playbooks/false-favorite-trap.md", "continental_cup:even_shallow")
            self.assertAlmostEqual(ev_ctx["p_0c_anchor"], 0.40, places=6)
            self.assertGreater(ev_ctx["prior_center"], 0.65)
            self.assertLess(ev_ctx["prior_center"], 0.75)

            # Verify "unknown" rows never pool into a fake 13th context bucket
            _MOD.bayesian_record_outcomes(
                self._make_ctx_obs(
                    "playbooks/false-favorite-trap.md",
                    [1, 1, 1],
                    p_baseline=0.50,
                    start_id=3910400,
                    league="",
                    league_tier="",
                    ah_tier_and_refund="",
                ),
                ledger_path=str(ledger),
            )
            loaded_unk, _ = _MOD.BayesianTrackerEngine.load_ledger(ledger)
            ev_unk = _MOD.evaluate_context(loaded_unk, "playbooks/false-favorite-trap.md", "unknown")
            self.assertEqual(ev_unk["n_obs"], 0)

    def test_ae2_bayes_ucb_residual_proves_drift_alert_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            # Primary leader
            _MOD.bayesian_record_outcomes(
                self._make_ctx_obs("playbooks/tree-a.md", [1] * 12 + [0] * 3, p_baseline=0.50, start_id=3920000),
                ledger_path=str(ledger),
            )
            # Healthy challenger with LOWER ucb_excess than DRIFT_ALERT tree-c
            _MOD.bayesian_record_outcomes(
                self._make_ctx_obs("playbooks/tree-b.md", [1] * 8 + [0] * 7, p_baseline=0.50, start_id=3920050),
                ledger_path=str(ledger),
            )
            # DRIFT_ALERT challenger: 25 wins then 10 losses -> higher ucb_excess than tree-b, MUST still be excluded!
            _MOD.bayesian_record_outcomes(
                self._make_ctx_obs("playbooks/tree-c.md", [1] * 25 + [0] * 10, p_baseline=0.50, start_id=3920100),
                ledger_path=str(ledger),
            )
            sel = _MOD.bayesian_select_playbooks(
                candidate_ids=["playbooks/tree-c.md", "playbooks/tree-b.md", "playbooks/tree-a.md", "H1"],
                context_bucket="top5:mid",
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
                mode="bayes_ucb",
            )
            self.assertEqual(sel["status"], "ok")
            by_id = {c["entity_id"]: c for c in sel["candidates"]}
            self.assertEqual(by_id["playbooks/tree-c.md"]["advisory_status_global"], "DRIFT_ALERT")
            self.assertGreater(by_id["playbooks/tree-c.md"]["ucb_excess"], by_id["playbooks/tree-b.md"]["ucb_excess"])
            self.assertEqual(sel["primary_seat"], "playbooks/tree-a.md")
            self.assertEqual(sel["residual_seat"], "H1")
            self.assertEqual(sel["shadow_candidates"], ["playbooks/tree-c.md", "playbooks/tree-b.md"])
            self.assertTrue(sel["exploration_triggered"])

            # Without H1, tree-c STILL beats tree-b on ucb_excess yet tree-b MUST win residual_seat
            sel_no_h1 = _MOD.bayesian_select_playbooks(
                candidate_ids=["playbooks/tree-c.md", "playbooks/tree-b.md", "playbooks/tree-a.md"],
                context_bucket="top5:mid",
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
                mode="bayes_ucb",
            )
            self.assertEqual(sel_no_h1["residual_seat"], "playbooks/tree-b.md")
            self.assertEqual(sel_no_h1["shadow_candidates"], ["playbooks/tree-c.md"])

            # Fallback when only DRIFT_ALERT challenger remains
            sel_fallback = _MOD.bayesian_select_playbooks(
                candidate_ids=["playbooks/tree-a.md", "playbooks/tree-c.md"],
                context_bucket="top5:mid",
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
                mode="bayes_ucb",
            )
            self.assertEqual(sel_fallback["residual_seat"], "playbooks/tree-c.md")
            self.assertFalse(sel_fallback["exploration_triggered"])

    def test_ae3_seeded_contextual_ts_determinism_and_input_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            _MOD.bayesian_record_outcomes(
                self._make_ctx_obs("playbooks/tree-a.md", [1] * 10 + [0] * 5, p_baseline=0.50, start_id=3930000),
                ledger_path=str(ledger),
            )
            args1 = dict(
                candidate_ids=["playbooks/tree-a.md", "H1", "H2"],
                context_bucket="top5:mid",
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
                mode="contextual_ts",
            )
            s1 = _MOD.bayesian_select_playbooks(**args1)
            s2 = _MOD.bayesian_select_playbooks(**args1)
            s_perm = _MOD.bayesian_select_playbooks(
                candidate_ids=["H2", "H1", "playbooks/tree-a.md"],
                context_bucket="top5:mid",
                cutoff_time="2026-12-01T12:00:00+08:00",
                ledger_path=str(ledger),
                mode="contextual_ts",
            )
            self.assertEqual(s1["mode"], "contextual_ts")
            self.assertEqual(s1["residual_seat"], s2["residual_seat"])
            self.assertEqual(s1["sampled_scores"], s2["sampled_scores"])
            self.assertEqual(s1["seed_digest"], s2["seed_digest"])
            self.assertEqual(s1["seed_digest"], s_perm["seed_digest"])
            self.assertEqual(s1["sampled_scores"], s_perm["sampled_scores"])
            self.assertEqual(s1["shadow_candidates"], s_perm["shadow_candidates"])

            sk1 = _MOD.bayesian_select_playbooks(**args1, seed_key="match-3999999")
            sk2 = _MOD.bayesian_select_playbooks(**args1, seed_key="match-3999999")
            self.assertEqual(sk1["seed_digest"], sk2["seed_digest"])
            self.assertEqual(sk1["sampled_scores"], sk2["sampled_scores"])
            self.assertNotEqual(sk1["seed_digest"], s1["seed_digest"])
            self.assertNotEqual(sk1["sampled_scores"], s1["sampled_scores"])
            by_id_cts = {c["entity_id"]: c for c in sk1["candidates"]}
            self.assertNotEqual(by_id_cts["H1"]["sampled_excess"], by_id_cts["H1"]["ucb_excess"])

            # API validation sad paths
            err_ctx = _MOD.bayesian_select_playbooks(candidate_ids=["H1"], context_bucket="bad:bucket", cutoff_time="2026-12-01T12:00:00+08:00", ledger_path=str(ledger))
            self.assertEqual(err_ctx["error"]["error_code"], "INVALID_CONTEXT_BUCKET")
            err_str = _MOD.bayesian_select_playbooks(candidate_ids="playbooks/tree-a.md", context_bucket="top5:mid", cutoff_time="2026-12-01T12:00:00+08:00", ledger_path=str(ledger))  # type: ignore[arg-type]
            self.assertEqual(err_str["error"]["error_code"], "INVALID_CANDIDATE_IDS")
            err_nan = _MOD.bayesian_select_playbooks(candidate_ids=["H1"], context_bucket="top5:mid", cutoff_time="2026-12-01T12:00:00+08:00", ledger_path=str(ledger), ucb_quantile=float("nan"))
            self.assertEqual(err_nan["error"]["error_code"], "INVALID_HYPERPARAMETER")

            s_one = _MOD.bayesian_select_playbooks(candidate_ids=["H1"], context_bucket="top5:mid", cutoff_time="2026-12-01T12:00:00+08:00", ledger_path=str(ledger))
            self.assertEqual(s_one["primary_seat"], "H1")
            self.assertIsNone(s_one["residual_seat"])

    def test_ae4_causal_cutoff_role_counts_and_health_replay_filter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = pathlib.Path(td) / "bayes.jsonl"
            cutoff = "2026-11-26T00:35:28+08:00"
            rows = []
            rows.extend(
                self._make_ctx_obs(
                    "playbooks/valid-tree.md",
                    [1],
                    start_id=3940000,
                    start_dt=datetime(2026, 11, 25, 20, 0, tzinfo=timezone(timedelta(hours=8))),
                    context_bucket="TOP5:MID",
                    observation_role="shadow",
                )
            )
            rows.extend(
                self._make_ctx_obs(
                    "H3",
                    [1],
                    start_id=3940010,
                    start_dt=datetime(2026, 11, 25, 22, 0, tzinfo=timezone(timedelta(hours=8))),
                    observation_role="residual",
                )
            )
            rows.extend(
                self._make_ctx_obs(
                    "H4",
                    [1],
                    start_id=3940020,
                    start_dt=datetime(2026, 11, 26, 4, 0, tzinfo=timezone(timedelta(hours=8))),
                )
            )
            rec_res = _MOD.bayesian_record_outcomes(rows, ledger_path=str(ledger))
            self.assertEqual(rec_res["status"], "ok")
            self.assertEqual(rec_res["added_count"], 3)

            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "match_id": "3940031", "sales_date": "2026-11-20",
                    "kickoff_at": "2026-11-20T20:00:00+08:00", "cutoff_time": "2026-11-20T17:00:00+08:00",
                    "entity_id": "H5", "entity_type": "draft", "p_baseline": 0.50, "hit": 1,
                    "context_bucket": "top5:mid", "observation_role": "primary", "known_answer_replay": True,
                }) + "\n")

            sel = _MOD.bayesian_select_playbooks(
                candidate_ids=["playbooks/valid-tree.md", "H3", "H4", "H5"],
                league="英超", league_tier="五大联赛", ah_tier_and_refund="中盘(-0.75) ; half_refund",
                cutoff_time=cutoff, ledger_path=str(ledger),
            )
            self.assertEqual(sel["causal_rows"], 1)
            by_id = {c["entity_id"]: c for c in sel["candidates"]}
            self.assertEqual(by_id["playbooks/valid-tree.md"]["n_global"], 1)
            self.assertEqual(by_id["H5"]["n_global"], 0)

            rep = _MOD.bayesian_health_report(ledger_path=str(ledger), context_bucket="top5:mid", include_contexts=True)
            self.assertEqual(rep["status"], "ok")
            self.assertEqual(rep["role_counts"], {"primary": 1, "residual": 1, "shadow": 1})
            eids = {e["entity_id"] for e in rep["entities"]}
            self.assertNotIn("H5", eids)
            ent = next(e for e in rep["entities"] if e["entity_id"] == "playbooks/valid-tree.md")
            self.assertIn("top5:mid", ent["contexts"])

            bad_rep = _MOD.bayesian_health_report(ledger_path=str(ledger), context_bucket="bogus:bucket")
            self.assertEqual(bad_rep["error"]["error_code"], "INVALID_CONTEXT_BUCKET")

    def test_bootstrap_from_repo_roundtrip_and_cli_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            day_dir = root / "data" / "2026-11-20"
            cards_dir = day_dir / "cards"
            cards_dir.mkdir(parents=True)
            (day_dir / "meta.md").write_text(
                "| 3950001 | 巴西甲 | full_time=2:1 ✅ |\n",
                encoding="utf-8",
            )
            card_yaml = (
                "announce:\n"
                "  memory_summary:\n"
                "    identity:\n"
                "      league: '巴西甲'\n"
                "      kickoffTime: '2026-11-20T20:00:00+08:00'\n"
                "    regime:\n"
                "      league_tier: '次级联赛'\n"
                "      ah_tier_and_refund: '平手 ; full_refund'\n"
                "    data_ref:\n"
                "      odds_cutoff: '2026-11-20T18:00:00+08:00'\n"
                "    decision:\n"
                "      primary: '胜'\n"
                "      probabilities: {home: 0.55, draw: 0.25, away: 0.20}\n"
                "    wiki_scan:\n"
                "      shadow_candidates:\n"
                "        - {page: 'playbooks/shadow-tree.md', direction: '客胜'}\n"
                "    wiki_hits:\n"
                "      - {role: 'primary', page: 'playbooks/main-tree.md', direction: '主胜'}\n"
                "      - {role: 'residual', page: 'H1', direction: '平'}\n"
            )
            (cards_dir / "3950001.yaml").write_text(card_yaml, encoding="utf-8")

            out_b = root / "data" / "bayesian_tracker.jsonl"
            proc = subprocess.run(
                [sys.executable, str(_SERVER_PATH), "--bootstrap-bayesian-ledger", "--repo-root", str(root), "--output", str(out_b)],
                capture_output=True, text=True, check=False, timeout=15,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("BAYES_LEDGER_OK 4 ", proc.stdout)

            ver = subprocess.run(
                [sys.executable, str(_SERVER_PATH), "--verify-bayesian-ledger", str(out_b), "--min-rows", "4"],
                capture_output=True, text=True, check=False, timeout=15,
            )
            self.assertEqual(ver.returncode, 0, ver.stdout)

            boot_rows = [json.loads(l) for l in out_b.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual({r["entity_id"]: r["observation_role"] for r in boot_rows}, {
                "portfolio:1x2_primary": "primary",
                "playbooks/main-tree.md": "primary",
                "H1": "residual",
                "playbooks/shadow-tree.md": "shadow",
            })
            self.assertTrue(all(r["context_bucket"] == "other_domestic:even_shallow" for r in boot_rows))

    def test_poisson_engine_and_batch_error_paths(self) -> None:
        poi = _MOD.poisson_dixon_coles(1.80, 0.90)
        self.assertEqual(len(poi["top_scores"]), 6)
        self.assertAlmostEqual(poi["expected_total_goals"], 2.7, places=2)
        err_batch = _MOD.devig_oo_epc_batch([{"cid": 177, "open": [0.95, 3.0, 3.0]}])
        self.assertEqual(err_batch["status"], "error")
        self.assertEqual(len(err_batch["errors"]), 1)

    def test_stdio_mcp_bayesian_select_playbooks_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            b_ledger = pathlib.Path(td) / "bayesian.jsonl"
            proc = subprocess.Popen(
                [sys.executable, str(_SERVER_PATH)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8",
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
                r2 = _rpc(
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "bayesian_record_outcomes", "arguments": {"records": self._make_ctx_obs("playbooks/tree-a.md", [1, 1, 1, 0, 1]), "ledger_path": str(b_ledger)}}},
                    expect_id=2,
                )
                r2_payload = json.loads(r2["result"]["content"][0]["text"])
                self.assertEqual(r2_payload["status"], "ok")
                self.assertEqual(r2_payload["added_count"], 5)

                r3 = _rpc(
                    {
                        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {
                            "name": "bayesian_select_playbooks",
                            "arguments": {
                                "candidate_ids": ["playbooks/tree-a.md", "H1", "H2"],
                                "context_bucket": "top5:mid",
                                "cutoff_time": "2026-12-01T12:00:00+08:00",
                                "ledger_path": str(b_ledger),
                            },
                        },
                    },
                    expect_id=3,
                )
                sel_payload = json.loads(r3["result"]["content"][0]["text"])
                self.assertEqual(sel_payload["status"], "ok")
                self.assertEqual(sel_payload["primary_seat"], "playbooks/tree-a.md")
                self.assertTrue(sel_payload["exploration_triggered"])
            finally:
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
                proc.terminate()
                proc.wait(timeout=5)

    def test_u1_bom_caller_order_double_chance_and_compact_health(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            # AE1: UTF-8 BOM tolerance on Conformal & Bayesian ledgers + BOM-free rewrite
            conf_ledger = root / "conformal.jsonl"
            bayes_ledger = root / "bayesian.jsonl"
            conf_row = {
                "match_id": "3960001", "sales_date": "2026-10-01",
                "kickoff_at": "2026-10-01T20:00:00+08:00", "cutoff_time": "2026-10-01T17:00:00+08:00",
                "league": "英超", "probs": [0.50, 0.25, 0.25], "outcome": "H", "source_card": "c1",
            }
            bayes_row = {
                "match_id": "3960001", "sales_date": "2026-10-01",
                "kickoff_at": "2026-10-01T20:00:00+08:00", "cutoff_time": "2026-10-01T17:00:00+08:00",
                "entity_id": "playbooks/tree-z.md", "entity_type": "playbook", "direction": "主胜",
                "p_baseline": 0.50, "hit": 1, "source_card": "c1",
                "league_tier": "top5", "ah_tier": "mid", "context_bucket": "top5:mid", "observation_role": "primary",
            }
            conf_ledger.write_text(json.dumps(conf_row, ensure_ascii=False) + "\n", encoding="utf-8-sig")
            bayes_ledger.write_text(json.dumps(bayes_row, ensure_ascii=False) + "\n", encoding="utf-8-sig")
            self.assertTrue(conf_ledger.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertTrue(bayes_ledger.read_bytes().startswith(b"\xef\xbb\xbf"))

            cp = _MOD.conformal_predict_1x2([0.5, 0.25, 0.25], cutoff_time="2026-11-01T12:00:00+08:00", ledger_path=str(conf_ledger))
            self.assertEqual(cp["status"], "insufficient_calibration")
            self.assertEqual(cp["n_cal"], 1)
            hr_bom = _MOD.bayesian_health_report(ledger_path=str(bayes_ledger))
            self.assertEqual(hr_bom["status"], "ok")
            self.assertEqual(hr_bom["total_rows"], 1)
            sel_bom = _MOD.bayesian_select_playbooks(
                candidate_ids=["playbooks/tree-z.md", "H1", "H2"],
                context_bucket="top5:mid", cutoff_time="2026-11-01T12:00:00+08:00", ledger_path=str(bayes_ledger),
            )
            self.assertEqual(sel_bom["status"], "ok")
            self.assertEqual(sel_bom["causal_rows"], 1)
            # Rewriting strips BOM
            _MOD.bayesian_record_outcomes(
                self._make_ctx_obs("playbooks/tree-y.md", [1], start_id=3960010),
                ledger_path=str(bayes_ledger),
            )
            self.assertFalse(bayes_ledger.read_bytes().startswith(b"\xef\xbb\xbf"))

            # AE2: Cold-start caller-order tie-break + telemetry + CTS seed invariance
            empty_ledger = root / "empty_bayes.jsonl"
            order1 = [
                "playbooks/price-over-handicap.md",
                "playbooks/draw-compression-lock.md",
                "playbooks/crushing-vig-suppression.md",
            ]
            s_o1 = _MOD.bayesian_select_playbooks(
                candidate_ids=order1, context_bucket="national_team:even_shallow",
                cutoff_time="2026-11-15T21:21:00+08:00", ledger_path=str(empty_ledger), mode="bayes_ucb",
            )
            self.assertEqual(s_o1["primary_seat"], order1[0])
            self.assertEqual(s_o1["residual_seat"], order1[1])
            self.assertEqual(s_o1["shadow_candidates"], [order1[2]])
            self.assertEqual(s_o1["tie_break_policy"], "caller_order")
            self.assertEqual(s_o1["tie_break_applied"], {"primary": True, "residual": True})

            order2 = list(reversed(order1))
            s_o2 = _MOD.bayesian_select_playbooks(
                candidate_ids=order2, context_bucket="national_team:even_shallow",
                cutoff_time="2026-11-15T21:21:00+08:00", ledger_path=str(empty_ledger), mode="bayes_ucb",
            )
            self.assertEqual(s_o2["primary_seat"], order2[0])
            self.assertEqual(s_o2["residual_seat"], order2[1])
            self.assertEqual(s_o2["shadow_candidates"], [order2[2]])
            cts1 = _MOD.bayesian_select_playbooks(
                candidate_ids=order1, context_bucket="national_team:even_shallow",
                cutoff_time="2026-11-15T21:21:00+08:00", ledger_path=str(empty_ledger), mode="contextual_ts", seed_key="2981563",
            )
            cts2 = _MOD.bayesian_select_playbooks(
                candidate_ids=order2, context_bucket="national_team:even_shallow",
                cutoff_time="2026-11-15T21:21:00+08:00", ledger_path=str(empty_ledger), mode="contextual_ts", seed_key="2981563",
            )
            self.assertEqual(cts1["seed_digest"], cts2["seed_digest"])
            self.assertEqual(cts1["sampled_scores"], cts2["sampled_scores"])

            # Non-tie statistical win beats caller order
            _MOD.bayesian_record_outcomes(
                self._make_ctx_obs("playbooks/winner.md", [1] * 8, p_baseline=0.45, start_id=3960100)
                + self._make_ctx_obs("playbooks/loser.md", [0] * 8, p_baseline=0.55, start_id=3960200),
                ledger_path=str(empty_ledger),
            )
            s_stat = _MOD.bayesian_select_playbooks(
                candidate_ids=["playbooks/loser.md", "playbooks/winner.md", "playbooks/fresh.md"],
                context_bucket="top5:mid", cutoff_time="2026-12-01T12:00:00+08:00", ledger_path=str(empty_ledger),
            )
            self.assertEqual(s_stat["primary_seat"], "playbooks/winner.md")
            self.assertEqual(s_stat["residual_seat"], "playbooks/fresh.md")
            self.assertEqual(s_stat["tie_break_applied"], {"primary": False, "residual": False})

            # AE3: Double-chance bootstrap + clamping + live UNSUPPORTED_DIRECTION guard
            rec_dc = _MOD.bayesian_record_outcomes(
                [
                    {
                        "match_id": "3960301", "sales_date": "2026-11-15",
                        "kickoff_at": "2026-11-15T22:00:00+08:00", "cutoff_time": "2026-11-15T21:00:00+08:00",
                        "entity_id": "playbooks/crushing-vig-suppression.md", "direction": "主不败",
                        "p_baseline": 0.70, "hit": 1, "context_bucket": "national_team:even_shallow",
                    },
                    {
                        "match_id": "3960302", "sales_date": "2026-11-15",
                        "kickoff_at": "2026-11-15T22:00:00+08:00", "cutoff_time": "2026-11-15T21:00:00+08:00",
                        "entity_id": "playbooks/price-over-handicap.md", "direction": "客不败",
                        "p_baseline": 0.55, "hit": 1, "context_bucket": "national_team:even_shallow",
                    },
                    {
                        "match_id": "3960303", "sales_date": "2026-11-15",
                        "kickoff_at": "2026-11-15T22:00:00+08:00", "cutoff_time": "2026-11-15T21:00:00+08:00",
                        "entity_id": "playbooks/false-favorite-trap.md", "direction": "受让不败",
                        "p_baseline": 0.55, "hit": 1, "context_bucket": "national_team:even_shallow",
                    },
                ],
                ledger_path=str(empty_ledger),
            )
            self.assertEqual(rec_dc["added_count"], 2)
            self.assertEqual(len(rec_dc["skipped"]), 1)
            self.assertEqual(rec_dc["skipped"][0]["reason"], "UNSUPPORTED_DIRECTION")

            repo = root / "dc_repo"
            cards_dir = repo / "data" / "2026-11-20" / "cards"
            cards_dir.mkdir(parents=True)
            (repo / "data" / "2026-11-20" / "meta.md").write_text(
                "| 3960401 | 欧国联 | full_time=1:1 ✅ |\n| 3960402 | 欧国联 | full_time=5:0 ✅ |\n",
                encoding="utf-8",
            )
            (cards_dir / "3960401.yaml").write_text(
                "announce:\n  memory_summary:\n    identity: {league: '欧国联', kickoffTime: '2026-11-20T20:00:00+08:00'}\n"
                "    regime: {league_tier: '国家队赛事', ah_tier_and_refund: '平半 ; half_refund'}\n"
                "    data_ref: {odds_cutoff: '2026-11-20T18:00:00+08:00'}\n"
                "    decision: {primary: '平', probabilities: {home: 0.45, draw: 0.25, away: 0.30}}\n"
                "    wiki_hits:\n"
                "      - {role: 'primary', page: 'playbooks/crushing-vig-suppression.md', direction: '主不败'}\n"
                "      - {role: 'residual', page: 'playbooks/price-over-handicap.md', direction: '客不败'}\n"
                "      - {role: 'residual', page: 'playbooks/false-favorite-trap.md', direction: '受让不败'}\n",
                encoding="utf-8",
            )
            (cards_dir / "3960402.yaml").write_text(
                "announce:\n  memory_summary:\n    identity: {league: '欧国联', kickoffTime: '2026-11-20T22:00:00+08:00'}\n"
                "    regime: {league_tier: '国家队赛事', ah_tier_and_refund: '深盘(-3.5) ; half_refund'}\n"
                "    data_ref: {odds_cutoff: '2026-11-20T18:00:00+08:00'}\n"
                "    decision: {primary: '胜', probabilities: {home: 0.958, draw: 0.040, away: 0.002}}\n"
                "    wiki_hits:\n"
                "      - {role: 'primary', page: 'playbooks/tiered-defense-mismatch.md', direction: '主不败'}\n",
                encoding="utf-8",
            )
            dc_out = repo / "data" / "bayesian_tracker.jsonl"
            n_dc, _ = _MOD.BayesianTrackerEngine.bootstrap_from_repo(str(repo), str(dc_out))
            self.assertEqual(n_dc, 5)
            dc_rows = {(r["match_id"], r["entity_id"]): r for r in _MOD.BayesianTrackerEngine.load_ledger(dc_out)[0]}
            self.assertAlmostEqual(dc_rows[("3960401", "playbooks/crushing-vig-suppression.md")]["p_baseline"], 0.70, places=6)
            self.assertEqual(dc_rows[("3960401", "playbooks/crushing-vig-suppression.md")]["hit"], 1)
            self.assertAlmostEqual(dc_rows[("3960401", "playbooks/price-over-handicap.md")]["p_baseline"], 0.55, places=6)
            self.assertEqual(dc_rows[("3960401", "playbooks/price-over-handicap.md")]["hit"], 1)
            self.assertNotIn(("3960401", "playbooks/false-favorite-trap.md"), dc_rows)
            self.assertAlmostEqual(dc_rows[("3960402", "playbooks/tiered-defense-mismatch.md")]["p_baseline"], 0.98, places=6)
            self.assertEqual(dc_rows[("3960402", "playbooks/tiered-defense-mismatch.md")]["hit"], 1)

            # AE4: Default compact health_report (no per-entity contexts, has context_counts) + context_bucket auto-drilldown
            rep_def = _MOD.bayesian_health_report(ledger_path=str(empty_ledger))
            self.assertEqual(rep_def["status"], "ok")
            self.assertIn("context_counts", rep_def)
            self.assertEqual(rep_def["context_counts"].get("top5:mid"), 16)
            self.assertEqual(rep_def["context_counts"].get("national_team:even_shallow"), 2)
            self.assertTrue(all("contexts" not in e for e in rep_def["entities"]))

            rep_bkt = _MOD.bayesian_health_report(ledger_path=str(empty_ledger), context_bucket="top5:mid")
            self.assertTrue(all("contexts" in e and list(e["contexts"].keys()) == ["top5:mid"] for e in rep_bkt["entities"]))

            rep_full = _MOD.bayesian_health_report(ledger_path=str(empty_ledger), include_contexts=True)
            self.assertTrue(all("contexts" in e for e in rep_full["entities"]))

            rep_empty_want = _MOD.bayesian_health_report(ledger_path=str(empty_ledger), entity_ids=[])
            self.assertEqual(rep_empty_want["status"], "ok")
            self.assertEqual(rep_empty_want["entity_count"], 0)
            self.assertEqual(rep_empty_want["role_counts"], {"primary": 0, "residual": 0, "shadow": 0})
            self.assertEqual(rep_empty_want["context_counts"], {})
            self.assertEqual(_MOD.bayesian_health_report(ledger_path=str(empty_ledger), entity_ids="H1")["error"]["error_code"], "INVALID_ENTITY_IDS")
            self.assertEqual(_MOD.bayesian_health_report(ledger_path=str(empty_ledger), context_bucket=123)["error"]["error_code"], "INVALID_CONTEXT_BUCKET")

            # Miss paths for double-chance (主不败 on away win 'A', 客不败 on home win 'H', 平局 alias)
            dc_live_miss = root / "dc_live_miss.jsonl"
            miss_rec = _MOD.bayesian_record_outcomes(
                ledger_path=str(dc_live_miss),
                records=[
                    {"match_id": "3960403", "sales_date": "2026-11-21", "kickoff_at": "2026-11-21T20:00:00+08:00", "cutoff_time": "2026-11-21T16:00:00+08:00", "entity_id": "playbooks/crushing-vig-suppression.md", "direction": "主不败", "p_baseline": 0.65, "hit": 0, "league": "英超", "league_tier": "top5", "ah_tier_and_refund": "平手"},
                    {"match_id": "3960404", "sales_date": "2026-11-21", "kickoff_at": "2026-11-21T20:00:00+08:00", "cutoff_time": "2026-11-21T16:00:00+08:00", "entity_id": "playbooks/price-over-handicap.md", "direction": "客不败", "p_baseline": 0.55, "hit": 0, "league": "英超", "league_tier": "top5", "ah_tier_and_refund": "平手"},
                    {"match_id": "3960405", "sales_date": "2026-11-21", "kickoff_at": "2026-11-21T20:00:00+08:00", "cutoff_time": "2026-11-21T16:00:00+08:00", "entity_id": "playbooks/draw-compression-lock.md", "direction": "平局", "p_baseline": 0.30, "hit": 1, "league": "英超", "league_tier": "top5", "ah_tier_and_refund": "平手"},
                ],
            )
            self.assertEqual(miss_rec["added_count"], 3)

            # DRIFT_ALERT candidate stays excluded from residual_seat and flagged in candidates while caller order governs tied healthy shadow candidates
            alert_ledger = root / "alert_shadow.jsonl"
            alert_rows = []
            for idx in range(10):
                dt = f"2026-11-{10 + idx:02d}"
                alert_rows.append({
                    "match_id": f"39605{idx:02d}", "sales_date": dt,
                    "kickoff_at": f"{dt}T20:00:00+08:00", "cutoff_time": f"{dt}T16:00:00+08:00",
                    "entity_id": "playbooks/head-fake.md", "direction": "主胜", "p_baseline": 0.70, "hit": 0,
                    "league": "英超", "league_tier": "top5", "ah_tier_and_refund": "主让(-0.75)",
                })
            _MOD.bayesian_record_outcomes(ledger_path=str(alert_ledger), records=alert_rows)
            sel_alert = _MOD.bayesian_select_playbooks(
                candidate_ids=["playbooks/crushing-vig-suppression.md", "playbooks/price-over-handicap.md", "playbooks/head-fake.md", "playbooks/draw-compression-lock.md"],
                league="英超", league_tier="top5", ah_tier_and_refund="主让(-0.75)",
                cutoff_time="2026-11-25T16:00:00+08:00", ledger_path=str(alert_ledger), mode="bayes_ucb",
            )
            self.assertEqual(sel_alert["status"], "ok")
            by_alert = {c["entity_id"]: c for c in sel_alert["candidates"]}
            self.assertEqual(by_alert["playbooks/head-fake.md"]["advisory_status_global"], "DRIFT_ALERT")
            self.assertNotEqual(sel_alert["residual_seat"], "playbooks/head-fake.md")
            self.assertEqual(sel_alert["shadow_candidates"], ["playbooks/draw-compression-lock.md", "playbooks/head-fake.md"])



if __name__ == "__main__":
    unittest.main()
