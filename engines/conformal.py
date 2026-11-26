"""Pure-stdlib Conformal Prediction (LAC + Mondrian + ACI + Inductive Venn-Abers) engine."""

from __future__ import annotations

import hashlib
import math
import pathlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .common import (
    BJ_TZ,
    CLASSES,
    _is_strict_finite_number,
    _is_truthy_flag,
    atomic_write_ledger as _atomic_write_ledger,
    clean_str,
    is_valid_sales_date,
    iter_settled_prematch_cards,
    load_ledger as _load_ledger,
    normalize_probs as _normalize_probs,
    parse_timestamp as _parse_timestamp,
    resolve_ledger_path as _resolve_ledger_path,
)


class ConformalEngine:
    """Pure-stdlib Conformal Prediction (LAC + Mondrian + ACI + Inductive Venn-Abers) for 1X2 Football Outcomes."""

    CLASSES: Tuple[str, str, str] = CLASSES
    BJ_TZ = BJ_TZ
    _iter_settled_prematch_cards = staticmethod(iter_settled_prematch_cards)

    @classmethod
    def parse_timestamp(cls, value: Any) -> Optional[datetime]:
        return _parse_timestamp(value)

    @classmethod
    def normalize_probs(cls, probs: Any, context: str = "probs") -> Tuple[Optional[List[float]], Optional[dict]]:
        return _normalize_probs(probs, context=context)

    @staticmethod
    def resolve_ledger_path(ledger_path: Optional[str]) -> Tuple[Optional[pathlib.Path], Optional[dict]]:
        return _resolve_ledger_path(ledger_path, "SPORTS_MATH_CONFORMAL_LEDGER", "data/conformal_calibration.jsonl")

    @staticmethod
    def load_ledger(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Optional[dict]]:
        return _load_ledger(path)

    @staticmethod
    def atomic_write_ledger(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
        _atomic_write_ledger(path, rows, prefix=".conformal_")

    @classmethod
    def sort_key(cls, row: Dict[str, Any]) -> Tuple[str, str, str]:
        dt = cls.parse_timestamp(row.get("kickoff_at"))
        ko_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else "9999-99-99T99:99:99Z"
        return (str(row.get("sales_date", "")), ko_iso, str(row.get("match_id", "")))

    @classmethod
    def _filter_causal(
        cls,
        rows: List[Dict[str, Any]],
        cutoff_dt: Optional[datetime],
        cutoff_sales_date: Optional[str],
    ) -> List[Dict[str, Any]]:
        cutoff_bj_date = cutoff_dt.astimezone(cls.BJ_TZ).strftime("%Y-%m-%d") if cutoff_dt else cutoff_sales_date
        out: List[Dict[str, Any]] = []
        for r in rows:
            if _is_truthy_flag(r.get("known_answer_replay")):
                continue
            outcome = r.get("outcome")
            if outcome not in cls.CLASSES:
                continue
            probs = r.get("probs")
            if not isinstance(probs, list) or len(probs) != 3:
                continue
            ko_dt = cls.parse_timestamp(r.get("kickoff_at"))
            row_sd = str(r.get("sales_date", ""))
            if cutoff_dt is not None:
                if ko_dt is not None:
                    if ko_dt > cutoff_dt - timedelta(hours=3):
                        continue
                elif not row_sd or not cutoff_bj_date or row_sd >= cutoff_bj_date:
                    continue
            elif cutoff_sales_date and (not row_sd or row_sd >= cutoff_sales_date):
                continue
            out.append(r)
        out.sort(key=cls.sort_key)
        return out

    @staticmethod
    def finite_sample_quantile(scores: List[float], alpha: float) -> float:
        sv = sorted(scores)
        n = len(sv)
        idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - alpha)) - 1))
        return sv[idx]

    @classmethod
    def lac_set(cls, probs: List[float], q_hat: float) -> List[str]:
        cset = [cls.CLASSES[k] for k in range(3) if (1.0 - probs[k]) <= q_hat + 1e-12]
        if not cset:
            best = max(range(3), key=lambda k: probs[k])
            cset = [cls.CLASSES[best]]
        return cset

    @classmethod
    def _pav_venn_abers_binary(cls, cal_pairs: List[Tuple[float, int]], test_score: float) -> Tuple[float, float]:
        """Inductive Venn-Abers via 1D Pool Adjacent Violators with Laplace m=1 smoothing."""
        base_grouped: Dict[float, List[float | int]] = {}
        for s, y in cal_pairs:
            grp = base_grouped.setdefault(float(s), [0.0, 0])
            grp[0] += float(y)
            grp[1] += 1
        t_score = float(test_score)
        base_grouped.setdefault(t_score, [0.0, 0])
        sorted_keys = sorted(base_grouped.keys())

        results: List[float] = []
        for y_hyp in (0, 1):
            stack: List[List[Any]] = []
            for s_val in sorted_keys:
                sum_y, cnt = base_grouped[s_val]
                if s_val == t_score:
                    stack.append([sum_y + float(y_hyp), cnt + 1, True])
                else:
                    stack.append([sum_y, cnt, False])
                while len(stack) >= 2:
                    prev_sum, prev_cnt, prev_test = stack[-2]
                    curr_sum, curr_cnt, curr_test = stack[-1]
                    if (prev_sum / prev_cnt) > (curr_sum / curr_cnt):
                        stack.pop()
                        stack[-1] = [prev_sum + curr_sum, prev_cnt + curr_cnt, prev_test or curr_test]
                    else:
                        break

            p_smooth = 0.5
            for blk_sum, blk_cnt, has_test in stack:
                if has_test:
                    p_smooth = (blk_sum + 1.0) / (blk_cnt + 2.0)
                    break
            results.append(p_smooth)

        return round(min(results), 6), round(max(results), 6)

    @classmethod
    def predict_1x2(
        cls,
        probs: Any,
        cutoff_time: Optional[str] = None,
        cutoff_sales_date: Optional[str] = None,
        alpha: float = 0.20,
        window_size: int = 150,
        ledger_path: Optional[str] = None,
        min_cal: int = 20,
        min_class_cal: int = 15,
        min_venn_abers: int = 50,
    ) -> dict:
        norm_probs, p_err = cls.normalize_probs(probs, context="probs")
        if p_err is not None:
            return {"status": "error", "algorithm": "LAC-Mondrian-ACI-VA", "error": p_err}

        if not cutoff_time and not cutoff_sales_date:
            return {
                "status": "error",
                "algorithm": "LAC-Mondrian-ACI-VA",
                "error": {
                    "error_code": "MISSING_CUTOFF",
                    "message": "Provide cutoff_time (ISO-8601) or cutoff_sales_date (YYYY-MM-DD) to enforce causal zero-leakage filtering.",
                },
            }

        cutoff_dt: Optional[datetime] = None
        if cutoff_time:
            cutoff_dt = cls.parse_timestamp(cutoff_time)
            if cutoff_dt is None:
                return {
                    "status": "error",
                    "algorithm": "LAC-Mondrian-ACI-VA",
                    "error": {
                        "error_code": "INVALID_CUTOFF_TIME",
                        "received": cutoff_time,
                        "message": f"Unable to parse cutoff_time={cutoff_time!r}.",
                    },
                }

        if cutoff_sales_date and not is_valid_sales_date(cutoff_sales_date):
            return {
                "status": "error",
                "algorithm": "LAC-Mondrian-ACI-VA",
                "error": {
                    "error_code": "INVALID_CUTOFF_SALES_DATE",
                    "received": cutoff_sales_date,
                    "message": "cutoff_sales_date must be YYYY-MM-DD.",
                },
            }

        alpha_target = max(0.05, min(0.50, float(alpha)))
        win_cap = max(20, int(window_size))
        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "algorithm": "LAC-Mondrian-ACI-VA", "error": l_err}
        assert lpath is not None

        all_rows, read_err = cls.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "algorithm": "LAC-Mondrian-ACI-VA", "error": read_err}

        causal_all = cls._filter_causal(all_rows, cutoff_dt, cutoff_sales_date)
        causal_rows = causal_all[-win_cap:]
        n_cal = len(causal_rows)

        alpha_t = alpha_target
        if causal_rows:
            last_next = causal_rows[-1].get("alpha_next")
            stored_target = causal_rows[-1].get("alpha_target")
            if (
                _is_strict_finite_number(stored_target)
                and abs(alpha_target - float(stored_target)) <= 1e-9
                and _is_strict_finite_number(last_next)
            ):
                alpha_t = max(0.10, min(0.40, float(last_next)))
        alpha_t = round(alpha_t, 6)

        cls_counts = {c: 0 for c in cls.CLASSES}
        for r in causal_rows:
            cls_counts[r["outcome"]] += 1

        if n_cal < min_cal:
            return {
                "status": "insufficient_calibration",
                "algorithm": "LAC-Mondrian-ACI-VA",
                "ledger_path": str(lpath),
                "n_cal": n_cal,
                "min_cal_required": min_cal,
                "alpha_target": alpha_target,
                "alpha_t": alpha_t,
                "q_hat_lac": None,
                "prediction_set": None,
                "set_size": None,
                "tier": None,
                "mondrian": {
                    "class_counts": cls_counts,
                    "class_cov": {c: None for c in cls.CLASSES},
                    "draw_under_coverage_alert": False,
                },
                "venn_abers": {
                    "status": "unready",
                    "min_required": min_venn_abers,
                    "note": "Marginal 1-vs-rest calibrated intervals (do not sum to 1.0).",
                    "intervals_1x2": None,
                },
            }

        assert norm_probs is not None
        scores = [1.0 - float(r["probs"][cls.CLASSES.index(r["outcome"])]) for r in causal_rows]
        q_hat = round(cls.finite_sample_quantile(scores, alpha_t), 6)
        pred_set = cls.lac_set(norm_probs, q_hat)
        set_size = len(pred_set)
        tier = {1: "HIGH_CONVICTION", 2: "DUAL_NEEDED", 3: "HIGH_ENTROPY_NO_PARLAY"}[set_size]

        cls_hits = {c: 0 for c in cls.CLASSES}
        for r, sc in zip(causal_rows, scores):
            if sc <= q_hat + 1e-12:
                cls_hits[r["outcome"]] += 1
        cls_cov = {
            c: round(cls_hits[c] / cls_counts[c], 4) if cls_counts[c] > 0 else None
            for c in cls.CLASSES
        }
        draw_alert = bool(
            cls_counts["D"] >= min_class_cal
            and cls_cov["D"] is not None
            and cls_cov["D"] < (1.0 - alpha_t - 0.15)
        )

        if n_cal >= min_venn_abers:
            intervals: Dict[str, List[float]] = {}
            for k, cname in enumerate(cls.CLASSES):
                pairs = [(float(r["probs"][k]), 1 if r["outcome"] == cname else 0) for r in causal_rows]
                lo, hi = cls._pav_venn_abers_binary(pairs, norm_probs[k])
                intervals[cname] = [lo, hi]
            va_block: Dict[str, Any] = {
                "status": "ok",
                "min_required": min_venn_abers,
                "note": "Marginal 1-vs-rest Laplace-smoothed PAV intervals in (0, 1); do not sum to 1.0.",
                "intervals_1x2": intervals,
            }
        else:
            va_block = {
                "status": "unready",
                "min_required": min_venn_abers,
                "note": "Marginal 1-vs-rest Laplace-smoothed PAV intervals require >= 50 causal samples.",
                "intervals_1x2": None,
            }

        return {
            "status": "ok",
            "algorithm": "LAC-Mondrian-ACI-VA",
            "ledger_path": str(lpath),
            "n_cal": n_cal,
            "window_size": win_cap,
            "alpha_target": alpha_target,
            "alpha_t": alpha_t,
            "q_hat_lac": q_hat,
            "prediction_set": pred_set,
            "set_size": set_size,
            "tier": tier,
            "mondrian": {
                "class_counts": cls_counts,
                "class_cov": cls_cov,
                "draw_under_coverage_alert": draw_alert,
            },
            "venn_abers": va_block,
        }

    @classmethod
    def _replay_aci_rows(
        cls,
        rows: List[Dict[str, Any]],
        alpha_target: float = 0.20,
        gamma: float = 0.05,
        min_cal: int = 20,
        window_size: int = 150,
    ) -> List[Dict[str, Any]]:
        """Sort rows chronologically and deterministically compute/preserve LAC & ACI fields."""
        sorted_rows = sorted(rows, key=cls.sort_key)
        alpha_curr = max(0.10, min(0.40, float(alpha_target)))
        out: List[Dict[str, Any]] = []
        prior_meta: List[Tuple[bool, Optional[datetime], str, float]] = []

        for raw in sorted_rows:
            row = dict(raw)
            norm_p, err = cls.normalize_probs(row.get("probs"), context="probs")
            if err is not None or norm_p is None:
                continue
            row["probs"] = [round(x, 6) for x in norm_p]
            outcome = row.get("outcome")
            if outcome not in cls.CLASSES:
                continue
            lac_sc = round(1.0 - row["probs"][cls.CLASSES.index(outcome)], 6)
            row["lac_score"] = lac_sc
            row["alpha_target"] = round(float(row.get("alpha_target") or alpha_target), 6)

            ko_dt = cls.parse_timestamp(row.get("kickoff_at"))
            row_sd = str(row.get("sales_date", ""))
            cutoff_bj_date = ko_dt.astimezone(cls.BJ_TZ).strftime("%Y-%m-%d") if ko_dt else row_sd
            deadline = (ko_dt - timedelta(hours=3)) if ko_dt is not None else None

            prior_scores: List[float] = []
            for is_replay, p_ko, p_sd, p_score in prior_meta:
                if is_replay:
                    continue
                if ko_dt is not None:
                    assert deadline is not None
                    if p_ko is not None:
                        if p_ko > deadline:
                            continue
                    elif not p_sd or not cutoff_bj_date or p_sd >= cutoff_bj_date:
                        continue
                elif row_sd and (not p_sd or p_sd >= row_sd):
                    continue
                prior_scores.append(p_score)
            prior_scores = prior_scores[-window_size:]

            if len(prior_scores) < min_cal:
                row["covered_lac"] = None
                row["alpha_used"] = round(float(alpha_target), 6)
                row["alpha_next"] = round(float(alpha_target), 6)
            else:
                frozen_set = row.get("frozen_prediction_set")
                if isinstance(frozen_set, list) and all(x in cls.CLASSES for x in frozen_set) and len(frozen_set) > 0:
                    used_a = row.get("frozen_alpha_t")
                    alpha_used = max(0.10, min(0.40, float(used_a))) if isinstance(used_a, (int, float)) else alpha_curr
                    covered = outcome in frozen_set
                else:
                    alpha_used = alpha_curr
                    q_hat = cls.finite_sample_quantile(prior_scores, alpha_used)
                    covered = outcome in cls.lac_set(row["probs"], q_hat)
                err_t = 0.0 if covered else 1.0
                alpha_next = max(0.10, min(0.40, alpha_used + float(gamma) * (float(alpha_target) - err_t)))
                row["covered_lac"] = covered
                row["alpha_used"] = round(alpha_used, 6)
                row["alpha_next"] = round(alpha_next, 6)
                alpha_curr = alpha_next

            out.append(row)
            prior_meta.append((_is_truthy_flag(row.get("known_answer_replay")), ko_dt, row_sd, lac_sc))
        return out

    @classmethod
    def record_outcomes(
        cls,
        records: Any,
        ledger_path: Optional[str] = None,
        alpha_target: float = 0.20,
        gamma: float = 0.05,
        min_cal: int = 20,
    ) -> dict:
        if not isinstance(records, list) or len(records) == 0:
            return {"status": "error", "error": {"error_code": "EMPTY_RECORDS", "message": "records must be a non-empty list."}}

        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "error": l_err}
        assert lpath is not None

        existing, read_err = cls.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "error": read_err}

        existing_ids = {str(r.get("match_id")) for r in existing if r.get("match_id") is not None}
        skipped: List[Dict[str, str]] = []
        valid_candidates: List[Dict[str, Any]] = []

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                skipped.append({"index": str(idx), "reason": "INVALID_RECORD_TYPE"})
                continue
            mid = clean_str(rec.get("match_id"))
            if not mid:
                skipped.append({"index": str(idx), "reason": "MISSING_MATCH_ID"})
                continue
            if _is_truthy_flag(rec.get("known_answer_replay")):
                skipped.append({"match_id": mid, "reason": "KNOWN_ANSWER_REPLAY_REJECTED"})
                continue
            if mid in existing_ids:
                skipped.append({"match_id": mid, "reason": "DUPLICATE_MATCH_ID"})
                continue
            outcome = str(rec.get("outcome", "")).strip().upper()
            if outcome not in cls.CLASSES:
                skipped.append({"match_id": mid, "reason": f"INVALID_OUTCOME:{outcome}"})
                continue
            norm_p, p_err = cls.normalize_probs(rec.get("probs"), context=f"{mid}.probs")
            if p_err is not None or norm_p is None:
                skipped.append({"match_id": mid, "reason": p_err["error_code"] if p_err else "INVALID_PROBS"})
                continue
            ko_str = str(rec.get("kickoff_at", "")).strip()
            ko_dt = cls.parse_timestamp(ko_str)
            if ko_dt is None:
                skipped.append({"match_id": mid, "reason": "INVALID_KICKOFF_AT"})
                continue
            sd_str = str(rec.get("sales_date", "")).strip()
            if not is_valid_sales_date(sd_str):
                skipped.append({"match_id": mid, "reason": "INVALID_SALES_DATE"})
                continue
            ct_dt = cls.parse_timestamp(rec.get("cutoff_time"))
            if ct_dt is None:
                skipped.append({"match_id": mid, "reason": "MISSING_OR_INVALID_CUTOFF"})
                continue
            if ct_dt >= ko_dt:
                skipped.append({"match_id": mid, "reason": "NON_PREMATCH_CUTOFF"})
                continue
            if "alpha_t" in rec and rec["alpha_t"] is not None and not _is_strict_finite_number(rec["alpha_t"]):
                skipped.append({"match_id": mid, "reason": "INVALID_ALPHA_T"})
                continue

            row_obj: Dict[str, Any] = {
                "match_id": mid,
                "sales_date": sd_str,
                "kickoff_at": ko_str,
                "league": clean_str(rec.get("league")),
                "probs": [round(x, 6) for x in norm_p],
                "outcome": outcome,
                "source_card": clean_str(rec.get("source_card")),
            }
            if isinstance(rec.get("prediction_set"), list):
                row_obj["frozen_prediction_set"] = [str(x).upper() for x in rec["prediction_set"] if str(x).upper() in cls.CLASSES]
            if _is_strict_finite_number(rec.get("alpha_t")):
                row_obj["frozen_alpha_t"] = round(float(rec["alpha_t"]), 6)

            existing_ids.add(mid)
            valid_candidates.append(row_obj)

        if not valid_candidates:
            last_alpha = existing[-1].get("alpha_next", alpha_target) if existing else alpha_target
            return {
                "status": "ok",
                "ledger_path": str(lpath),
                "added_count": 0,
                "skipped": skipped,
                "total_rows": len(existing),
                "latest_alpha_t": last_alpha,
            }

        replayed = cls._replay_aci_rows(existing + valid_candidates, alpha_target=alpha_target, gamma=gamma, min_cal=min_cal)
        cls.atomic_write_ledger(lpath, replayed)

        added_ids = {r["match_id"] for r in valid_candidates}
        accepted = [r for r in replayed if r["match_id"] in added_ids]
        scored_rows = [r for r in replayed if r.get("covered_lac") is not None]
        cov_rate = round(sum(1 for r in scored_rows if r["covered_lac"]) / len(scored_rows), 4) if scored_rows else None
        latest_alpha = replayed[-1].get("alpha_next", alpha_target) if replayed else alpha_target

        return {
            "status": "ok",
            "ledger_path": str(lpath),
            "added_count": len(accepted),
            "skipped": skipped,
            "total_rows": len(replayed),
            "latest_alpha_t": latest_alpha,
            "empirical_lac_coverage": cov_rate,
            "scored_rows": len(scored_rows),
        }

    @classmethod
    def verify_ledger_file(cls, ledger_path: str, min_rows: int = 50) -> Tuple[bool, str]:
        p = pathlib.Path(ledger_path)
        rows, err = cls.load_ledger(p)
        if err is not None:
            return False, f"ERR {err}"
        if len(rows) < min_rows:
            return False, f"ERR row_count={len(rows)} < min_rows={min_rows}"
        seen_ids = set()
        prev_key: Optional[Tuple[str, str, str]] = None
        for idx, r in enumerate(rows):
            mid = str(r.get("match_id", ""))
            if not mid or mid in seen_ids:
                return False, f"ERR duplicate_or_empty_match_id at row {idx}: {mid!r}"
            seen_ids.add(mid)
            if r.get("outcome") not in cls.CLASSES:
                return False, f"ERR invalid outcome at {mid}: {r.get('outcome')!r}"
            _, p_err = cls.normalize_probs(r.get("probs"), context=mid)
            if p_err is not None:
                return False, f"ERR invalid probs at {mid}: {p_err}"
            if idx < 20 and r.get("covered_lac") is not None:
                return False, f"ERR expected covered_lac=null for early row {idx} ({mid})"
            key = cls.sort_key(r)
            if prev_key is not None and key < prev_key:
                return False, f"ERR unsorted ledger at {mid}: {key} < {prev_key}"
            prev_key = key
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return True, f"LEDGER_OK {len(rows)} sha256={digest}"

    @classmethod
    def bootstrap_from_repo(cls, repo_root: str, output_path: str) -> Tuple[int, str]:
        """Deterministically build conformal calibration JSONL from pre-match YAML cards + meta/journal scores."""
        records: List[Dict[str, Any]] = []
        for mid, sales_date, ko_dt, _oc_dt, ident, _dec, _ms, norm_p, outcome in iter_settled_prematch_cards(repo_root):
            records.append({
                "match_id": mid,
                "sales_date": sales_date,
                "kickoff_at": ko_dt.astimezone(cls.BJ_TZ).isoformat(),
                "league": str(ident.get("league", "")).strip(),
                "probs": [round(x, 6) for x in norm_p],
                "outcome": outcome,
                "source_card": f"data/{sales_date}/cards/{mid}.yaml",
            })

        replayed = cls._replay_aci_rows(records, alpha_target=0.20, gamma=0.05, min_cal=20)
        out_p = pathlib.Path(output_path)
        cls.atomic_write_ledger(out_p, replayed)
        ok, msg = cls.verify_ledger_file(str(out_p), min_rows=0)
        if not ok:
            raise RuntimeError(f"Bootstrap failed verification: {msg}")
        return len(replayed), msg
