"""
sports-math MCP Server
Official Model Context Protocol server for sports analytics & betting mathematics.
Exposes:
- devig_oo_epc: Tokyo University Goto (2026) OO-EPC de-vigging convex optimization
- devig_oo_epc_batch: Single-call batch OO-EPC de-vigging across up to 21 bookmakers with Pinnacle scissor-gap comparison
- poisson_dixon_coles: Bivariate Poisson score matrix with low-score adjustment
- conformal_predict_1x2: Deterministic LAC + Mondrian draw alert + Laplace-smoothed Inductive Venn-Abers over strict causal history
- conformal_record_outcomes: Idempotent post-match ledger appender with Gibbs & Candes (2021) ACI alpha_t update
- bayesian_record_outcomes: Idempotent post-match (match_id, entity_id) Bayesian observation ledger appender
- bayesian_health_report: Market-anchored Beta posteriors, anytime-valid e-processes, discounted drift, and Page-CUSUM evaluator
"""

import argparse
import hashlib
import math
import os
import pathlib
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import json

try:
    # Official MCP SDK v2+ (where FastMCP is renamed to MCPServer)
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        # Standalone FastMCP (Prefect)
        from fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError:
        # Official MCP SDK v1.x fallback
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

mcp = FastMCP(
    name="sports-math",
    description="Mathematical computation engine for sports betting odds devigging, score probability modeling, conformal uncertainty calibration, and sequential Bayesian/e-process tracking.",
    version="1.3.0"
)


def _is_truthy_flag(val: Any) -> bool:
    """Return True if val is boolean True or a truthy replay string ('true', '1', 'yes')."""
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in ("true", "1", "yes"):
        return True
    return False


def _is_strict_finite_number(val: Any) -> bool:
    """Return True only for finite real numbers (explicitly rejecting bool, NaN, inf)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _betacf(a: float, b: float, x: float) -> float:
    """Modified Lentz continued fraction for incomplete beta I_x(a, b)."""
    maxit = 200
    eps = 3.0e-12
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Evaluate regularized incomplete beta I_x(a, b) = P(Beta(a, b) <= x) using pure stdlib."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return max(0.0, min(1.0, bt * _betacf(a, b, x) / a))
    return max(0.0, min(1.0, 1.0 - bt * _betacf(b, a, 1.0 - x) / b))


def beta_ppf(q: float, a: float, b: float, iters: int = 60) -> float:
    """Invert regularized incomplete beta I_x(a, b) = q via monotonic bisection."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class OoEpcCalculator:
    """Tokyo University Shota Goto (2026) OO-EPC De-vigging Algorithm"""

    @staticmethod
    def validate_triple(odds: List[Any], context: str = "odds") -> Tuple[Optional[List[float]], Optional[dict]]:
        if not isinstance(odds, (list, tuple)) or len(odds) != 3:
            return None, {
                "error_code": "INVALID_TRIPLE_LENGTH",
                "field": context,
                "received": odds,
                "message": f"{context} must be a 3-element list [home_odds, draw_odds, away_odds], got {odds!r}. Retry with exactly 3 decimal odds >= 1.0."
            }
        parsed: List[float] = []
        labels = ("home_odds", "draw_odds", "away_odds")
        for idx, val in enumerate(odds):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                return None, {
                    "error_code": "NON_NUMERIC_ODD",
                    "field": f"{context}.{labels[idx]}",
                    "received": val,
                    "message": f"{context}.{labels[idx]} must be a numeric decimal odd >= 1.0, got {val!r}."
                }
            if not math.isfinite(fval) or fval < 1.0:
                return None, {
                    "error_code": "ODD_OUT_OF_RANGE",
                    "field": f"{context}.{labels[idx]}",
                    "received": fval,
                    "message": f"{context}.{labels[idx]} must be >= 1.0 (received {fval}). Check column order (do not pass return-rate or handicap line as 1X2 odds)."
                }
            parsed.append(fval)
        return parsed, None

    @staticmethod
    def calculate(odds: List[float], total: float = 1.0) -> List[float]:
        inverses: List[float] = [1.0 / o for o in odds]
        booksum: float = sum(inverses)

        ses: List[float] = [math.sqrt(max(0.0, 1.0 - p)) for p in inverses]
        sum_se: float = sum(ses)

        if sum_se > 0.0 and booksum > total:
            step: float = (booksum - total) / sum_se
            output_probs: List[float] = [p - (s * step) for p, s in zip(inverses, ses)]
        else:
            output_probs = [0.0] * len(inverses)

        # Goto (2026) Alg. 5 Step 6: fall back to Alg. 1 multiplicative scaling when any
        # output hits <= 0 (incl. IEEE-754 near-zero from 1.00 floor odds) or booksum <= total.
        if any(p <= 1e-12 for p in output_probs) or (booksum <= total):
            normalizer = booksum / total
            output_probs = [p / normalizer for p in inverses]

        return output_probs

    @classmethod
    def solve_triple(cls, odds: List[float]) -> dict:
        probs = cls.calculate(odds)
        h_pct = probs[0] * 100.0
        d_pct = probs[1] * 100.0
        a_pct = probs[2] * 100.0
        booksum = sum(1.0 / o for o in odds)
        margin = booksum - 1.0
        return {
            "algorithm": "Goto-OO-EPC",
            "odds": odds,
            "probabilities": [round(p, 6) for p in probs],
            "percentages": [f"{h_pct:.2f}%", f"{d_pct:.2f}%", f"{a_pct:.2f}%"],
            "display_text": f"{h_pct:.1f}% / {d_pct:.1f}% / {a_pct:.1f}%",
            "margin": round(margin, 4),
            "margin_pct": round(margin * 100.0, 2),
            "prob_1x2": {
                "home": round(probs[0], 6),
                "draw": round(probs[1], 6),
                "away": round(probs[2], 6)
            },
            "percentages_1x2": {
                "home": f"{h_pct:.2f}%",
                "draw": f"{d_pct:.2f}%",
                "away": f"{a_pct:.2f}%"
            }
        }


class PoissonEngine:
    """Bivariate Poisson with Dixon-Coles Low-Score Factor Adjustments"""

    DEFAULT_LOW_FACTORS: Dict[Tuple[int, int], float] = {
        (0, 0): 1.10,
        (1, 0): 1.05,
        (0, 1): 1.05,
        (1, 1): 1.10,
    }

    @staticmethod
    def _pmf(lamb: float, k: int) -> float:
        return math.exp(-lamb) * (lamb ** k) / math.factorial(k)

    @classmethod
    def calculate(
        cls,
        home_lambda: float,
        away_lambda: float,
        max_goals: int = 8,
        top_n: int = 6,
    ) -> dict:
        l1 = max(0.4, min(3.5, float(home_lambda)))
        l2 = max(0.4, min(3.5, float(away_lambda)))

        raw_scores: List[Tuple[int, int, float]] = []
        for i in range(max_goals):
            for j in range(max_goals):
                p = cls._pmf(l1, i) * cls._pmf(l2, j)
                p *= cls.DEFAULT_LOW_FACTORS.get((i, j), 1.0)
                raw_scores.append((i, j, p))

        total_p = sum(p for _, _, p in raw_scores)
        norm_scores = [(i, j, p / total_p) for i, j, p in raw_scores]
        norm_scores.sort(key=lambda x: x[2], reverse=True)

        home_win_prob = sum(p for i, j, p in norm_scores if i > j)
        draw_prob = sum(p for i, j, p in norm_scores if i == j)
        away_win_prob = sum(p for i, j, p in norm_scores if i < j)

        top_list = []
        for i, j, p in norm_scores[:top_n]:
            fair = 1.0 / p if p > 0 else 999.0
            top_list.append({
                "score": f"{i}:{j}",
                "probability": round(p, 6),
                "percentage": f"{p * 100:.2f}%",
                "fair_odds": round(fair, 2)
            })

        return {
            "home_lambda": round(l1, 3),
            "away_lambda": round(l2, 3),
            "expected_total_goals": round(l1 + l2, 2),
            "distribution_1x2": {
                "home_win": f"{home_win_prob * 100:.2f}%",
                "draw": f"{draw_prob * 100:.2f}%",
                "away_win": f"{away_win_prob * 100:.2f}%"
            },
            "top_scores": top_list
        }


class ConformalEngine:
    """Pure-stdlib Conformal Prediction (LAC + Mondrian + ACI + Inductive Venn-Abers) for 1X2 Football Outcomes."""

    CLASSES: Tuple[str, str, str] = ("H", "D", "A")
    BJ_TZ = timezone(timedelta(hours=8))
    _TS_RE = re.compile(
        r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2})(?::(\d{2}))?)?(Z|[+-]\d{2}:?\d{2})?"
    )

    @classmethod
    def parse_timestamp(cls, value: Any) -> Optional[datetime]:
        """Extract the first timestamp from a string, defaulting missing tz to +08:00, and return UTC datetime."""
        if not value:
            return None
        match = cls._TS_RE.search(str(value).strip())
        if not match:
            return None
        date_part, hm_part, sec_part, tz_part = match.groups()
        hm_part = hm_part or "00:00"
        sec_part = sec_part or "00"
        base = datetime.strptime(f"{date_part} {hm_part}:{sec_part}", "%Y-%m-%d %H:%M:%S")
        if not tz_part:
            tz = cls.BJ_TZ
        elif tz_part == "Z":
            tz = timezone.utc
        else:
            tz_clean = tz_part.replace(":", "")
            sign = 1 if tz_clean[0] == "+" else -1
            hrs = int(tz_clean[1:3])
            mins = int(tz_clean[3:5]) if len(tz_clean) >= 5 else 0
            tz = timezone(timedelta(hours=sign * hrs, minutes=sign * mins))
        return base.replace(tzinfo=tz).astimezone(timezone.utc)

    @classmethod
    def normalize_probs(cls, probs: Any, context: str = "probs") -> Tuple[Optional[List[float]], Optional[dict]]:
        """Validate 3-class probabilities (0..1 or 0..100 percentages) and normalize to sum 1.0."""
        if not isinstance(probs, (list, tuple)) or len(probs) != 3:
            return None, {
                "error_code": "INVALID_PROBS_LENGTH",
                "field": context,
                "received": probs,
                "message": f"{context} must be a 3-element list [p_home, p_draw, p_away], got {probs!r}."
            }
        vals: List[float] = []
        for idx, raw in enumerate(probs):
            try:
                fval = float(raw)
            except (TypeError, ValueError):
                return None, {
                    "error_code": "NON_NUMERIC_PROB",
                    "field": f"{context}[{idx}]",
                    "received": raw,
                    "message": f"{context}[{idx}] must be numeric, got {raw!r}."
                }
            if not math.isfinite(fval) or fval <= 0.0:
                return None, {
                    "error_code": "PROB_OUT_OF_RANGE",
                    "field": f"{context}[{idx}]",
                    "received": fval,
                    "message": f"{context}[{idx}] must be strictly positive, got {fval}."
                }
            vals.append(fval)
        total = sum(vals)
        if total > 2.0:
            vals = [v / 100.0 for v in vals]
            total = sum(vals)
        if total < 0.85 or total > 1.15:
            return None, {
                "error_code": "PROBS_SUM_OUT_OF_RANGE",
                "field": context,
                "received": vals,
                "message": f"Normalized sum of {context} ({total:.4f}) is outside [0.85, 1.15]."
            }
        return [v / total for v in vals], None

    @staticmethod
    def resolve_ledger_path(ledger_path: Optional[str]) -> Tuple[Optional[pathlib.Path], Optional[dict]]:
        """Resolve ledger path; reject unanchored relative paths when env fallback is unset."""
        env_path = os.environ.get("SPORTS_MATH_CONFORMAL_LEDGER", "").strip()
        raw = (ledger_path or "").strip()
        if not raw:
            if not env_path:
                return None, {
                    "error_code": "INVALID_LEDGER_PATH",
                    "message": "ledger_path is required when SPORTS_MATH_CONFORMAL_LEDGER is unset. Pass an absolute path to data/conformal_calibration.jsonl."
                }
            p = pathlib.Path(env_path)
            if not p.is_absolute():
                return None, {
                    "error_code": "INVALID_LEDGER_PATH",
                    "message": f"SPORTS_MATH_CONFORMAL_LEDGER must be an absolute path, got {env_path!r}."
                }
            return p, None
        p = pathlib.Path(raw)
        if not p.is_absolute():
            if env_path and pathlib.Path(env_path).is_absolute():
                return pathlib.Path(env_path), None
            return None, {
                "error_code": "INVALID_LEDGER_PATH",
                "received": raw,
                "message": f"ledger_path must be an absolute path (got relative {raw!r}) when SPORTS_MATH_CONFORMAL_LEDGER is unset."
            }
        return p, None

    @staticmethod
    def load_ledger(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Optional[dict]]:
        """Load JSONL calibration ledger; empty list if file does not exist yet."""
        if not path.exists():
            return [], None
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError as exc:
                        return [], {
                            "error_code": "MALFORMED_LEDGER_LINE",
                            "line": lineno,
                            "message": f"Invalid JSON at {path}:{lineno}: {exc}"
                        }
                    if isinstance(obj, dict):
                        rows.append(obj)
        except OSError as exc:
            return [], {"error_code": "LEDGER_READ_ERROR", "message": str(exc)}
        return rows, None

    @staticmethod
    def atomic_write_ledger(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
        """Atomically persist sorted JSONL rows via same-directory temp file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".conformal_", suffix=".jsonl.tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

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
                else:
                    if not row_sd or not cutoff_bj_date or row_sd >= cutoff_bj_date:
                        continue
            elif cutoff_sales_date:
                if not row_sd or row_sd >= cutoff_sales_date:
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
        results: List[float] = []
        for y_hyp in (0, 1):
            pts = list(cal_pairs) + [(test_score, y_hyp)]
            # Group identical scores ascending; track whether the group holds the augmented test point
            grouped: Dict[float, List[Any]] = {}
            for s, y in pts[:-1]:
                grp = grouped.setdefault(float(s), [0.0, 0, False])
                grp[0] += float(y)
                grp[1] += 1
            t_grp = grouped.setdefault(float(test_score), [0.0, 0, False])
            t_grp[0] += float(y_hyp)
            t_grp[1] += 1
            t_grp[2] = True

            stack: List[List[Any]] = []
            for s_val in sorted(grouped.keys()):
                sum_y, cnt, has_test = grouped[s_val]
                stack.append([sum_y, cnt, has_test])
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
                    "message": "Provide cutoff_time (ISO-8601) or cutoff_sales_date (YYYY-MM-DD) to enforce causal zero-leakage filtering."
                }
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
                        "message": f"Unable to parse cutoff_time={cutoff_time!r}."
                    }
                }

        if cutoff_sales_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(cutoff_sales_date).strip()):
            return {
                "status": "error",
                "algorithm": "LAC-Mondrian-ACI-VA",
                "error": {
                    "error_code": "INVALID_CUTOFF_SALES_DATE",
                    "received": cutoff_sales_date,
                    "message": "cutoff_sales_date must be YYYY-MM-DD."
                }
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

        # Latest ACI alpha_t from the most recent causal row's alpha_next only if alpha_target matches
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
                    "draw_under_coverage_alert": False
                },
                "venn_abers": {
                    "status": "unready",
                    "min_required": min_venn_abers,
                    "note": "Marginal 1-vs-rest calibrated intervals (do not sum to 1.0).",
                    "intervals_1x2": None
                }
            }

        scores = [1.0 - float(r["probs"][cls.CLASSES.index(r["outcome"])]) for r in causal_rows]
        q_hat = round(cls.finite_sample_quantile(scores, alpha_t), 6)
        pred_set = cls.lac_set(norm_probs, q_hat)
        set_size = len(pred_set)
        tier_map = {1: "HIGH_CONVICTION", 2: "DUAL_NEEDED", 3: "HIGH_ENTROPY_NO_PARLAY"}
        tier = tier_map[set_size]

        # Mondrian class-conditional empirical coverage under current q_hat
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

        # Inductive Venn-Abers (Laplace-smoothed marginal 1-vs-rest intervals)
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
                "intervals_1x2": intervals
            }
        else:
            va_block = {
                "status": "unready",
                "min_required": min_venn_abers,
                "note": "Marginal 1-vs-rest Laplace-smoothed PAV intervals require >= 50 causal samples.",
                "intervals_1x2": None
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
                "draw_under_coverage_alert": draw_alert
            },
            "venn_abers": va_block
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
        for idx, raw in enumerate(sorted_rows):
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

            # Determine strictly earlier causal window (< row.kickoff_at - 3h, or < row.sales_date)
            ko_dt = cls.parse_timestamp(row.get("kickoff_at"))
            row_sd = str(row.get("sales_date", ""))
            prior = cls._filter_causal(out, ko_dt, row_sd)[-window_size:]

            if len(prior) < min_cal:
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
                    prior_scores = [float(pr["lac_score"]) for pr in prior]
                    q_hat = cls.finite_sample_quantile(prior_scores, alpha_used)
                    pred_set = cls.lac_set(row["probs"], q_hat)
                    covered = outcome in pred_set
                err_t = 0.0 if covered else 1.0
                alpha_next = max(0.10, min(0.40, alpha_used + float(gamma) * (float(alpha_target) - err_t)))
                row["covered_lac"] = covered
                row["alpha_used"] = round(alpha_used, 6)
                row["alpha_next"] = round(alpha_next, 6)
                alpha_curr = alpha_next
            out.append(row)
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
            return {
                "status": "error",
                "error": {"error_code": "EMPTY_RECORDS", "message": "records must be a non-empty list."}
            }

        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "error": l_err}
        assert lpath is not None

        existing, read_err = cls.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "error": read_err}

        existing_ids = {str(r.get("match_id")) for r in existing if r.get("match_id") is not None}
        accepted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []

        # Sort incoming deterministically first
        valid_candidates: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                skipped.append({"index": str(idx), "reason": "INVALID_RECORD_TYPE"})
                continue
            mid = str(rec.get("match_id", "")).strip()
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
            if not sd_str or not re.match(r"^\d{4}-\d{2}-\d{2}$", sd_str):
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
                "league": str(rec.get("league", "")).strip(),
                "probs": [round(x, 6) for x in norm_p],
                "outcome": outcome,
                "source_card": str(rec.get("source_card", "")).strip(),
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
                "latest_alpha_t": last_alpha
            }

        combined = existing + valid_candidates
        replayed = cls._replay_aci_rows(combined, alpha_target=alpha_target, gamma=gamma, min_cal=min_cal)
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
            "scored_rows": len(scored_rows)
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
    def _iter_settled_prematch_cards(cls, repo_root: str):
        """Yield validated settled prematch card tuples shared by ConformalEngine and BayesianTrackerEngine."""
        try:
            import yaml  # Lazy import so online MCP stays 100% stdlib
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required only for --bootstrap-* (install with `pip install 'oo-epc-mcp[bootstrap]'`)."
            ) from exc

        root = pathlib.Path(repo_root)
        data_dir = root / "data"
        seen_ids = set()

        for day_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name)):
            cards_dir = day_dir / "cards"
            if not cards_dir.is_dir():
                continue
            sales_date = day_dir.name
            scores_map: Dict[str, Tuple[int, int]] = {}
            for src_name in ("meta.md", "journal.md"):
                src_file = day_dir / src_name
                if not src_file.is_file():
                    continue
                txt = src_file.read_text(encoding="utf-8", errors="ignore")
                for line in txt.splitlines():
                    if not (line.startswith("|") or "[SCORE-PROVENANCE-GATE]" in line):
                        continue
                    m_id = re.search(r"\b(\d{7})\b", line)
                    if not m_id:
                        continue
                    mid = m_id.group(1)
                    if mid in scores_map:
                        continue
                    m_ft = re.search(r"full_time=(\d{1,2}):(\d{1,2})", line)
                    if m_ft:
                        scores_map[mid] = (int(m_ft.group(1)), int(m_ft.group(2)))
                        continue
                    if any(k in line for k in ("完场", "✅", "❌")):
                        for m_sc in re.finditer(r"(?<!\d)(\d{1,2}):(\d{1,2})(?!\d)", line):
                            h, a = int(m_sc.group(1)), int(m_sc.group(2))
                            before = line[max(0, m_sc.start() - 6):m_sc.start()]
                            if h <= 12 and a <= 12 and not re.search(r"\d{2}-\d{2}\s*$", before):
                                scores_map[mid] = (h, a)
                                break

            for card_path in sorted(cards_dir.glob("*.yaml")):
                mid = card_path.stem
                if mid in seen_ids or mid not in scores_map:
                    continue
                doc = yaml.safe_load(card_path.read_text(encoding="utf-8", errors="replace"))
                if not isinstance(doc, dict):
                    continue
                ann = doc.get("announce") or {}
                ca = ann.get("comprehensive_assessment") or {}
                ms = ann.get("memory_summary") or {}
                ident = ms.get("identity") or {}
                dref = ms.get("data_ref") or {}
                dec = ms.get("decision") or {}
                res = dref.get("research") or {}

                if (
                    _is_truthy_flag(doc.get("known_answer_replay"))
                    or _is_truthy_flag(ms.get("known_answer_replay"))
                    or _is_truthy_flag(dref.get("known_answer_replay"))
                ):
                    continue
                phase = str(res.get("analysis_phase", "prematch")).strip().lower()
                if phase not in ("", "prematch"):
                    continue

                ko_dt = cls.parse_timestamp(str(ident.get("kickoffTime", "")).strip())
                oc_dt = cls.parse_timestamp(str(dref.get("odds_cutoff", "")).strip())
                if ko_dt is None or oc_dt is None or oc_dt >= ko_dt:
                    continue

                probs_raw: Optional[List[float]] = None
                p_block = dec.get("probabilities")
                if isinstance(p_block, dict) and all(k in p_block for k in ("home", "draw", "away")):
                    probs_raw = [p_block["home"], p_block["draw"], p_block["away"]]
                else:
                    oo_str = str(ca.get("oo_epc_probabilities", ""))
                    m_oo = re.search(r"胜\s*([0-9.]+)%\s*/\s*平\s*([0-9.]+)%\s*/\s*负\s*([0-9.]+)%", oo_str)
                    if m_oo:
                        probs_raw = [float(m_oo.group(1)), float(m_oo.group(2)), float(m_oo.group(3))]
                if probs_raw is None:
                    continue
                norm_p, p_err = cls.normalize_probs(probs_raw, context=mid)
                if p_err is not None or norm_p is None:
                    continue

                hg, ag = scores_map[mid]
                outcome = "H" if hg > ag else ("D" if hg == ag else "A")
                seen_ids.add(mid)
                yield mid, sales_date, ko_dt, oc_dt, ident, dec, ms, norm_p, outcome

    @classmethod
    def bootstrap_from_repo(cls, repo_root: str, output_path: str) -> Tuple[int, str]:
        """Deterministically build conformal calibration JSONL from pre-match YAML cards + meta/journal scores."""
        records: List[Dict[str, Any]] = []
        for mid, sales_date, ko_dt, _oc_dt, ident, _dec, _ms, norm_p, outcome in cls._iter_settled_prematch_cards(repo_root):
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


class BayesianTrackerEngine:
    """Pure-stdlib Market-Anchored Beta Updating, Mixture E-Process, and Page-CUSUM Drift Tracker."""

    LAMBDA_GRID: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50)
    P_MIN: float = 0.02
    P_MAX: float = 0.98
    DEFAULT_PRIOR_STRENGTH: float = 4.0
    DEFAULT_DECAY_GAMMA: float = 0.95
    VALID_ENTITY_TYPES = ("draft", "playbook", "feature", "portfolio")
    _ENTITY_PATTERNS = (
        re.compile(r"^playbooks/[a-z0-9][a-z0-9-]*\.md$"),
        re.compile(r"^features/[a-z0-9][a-z0-9-]*\.md$"),
        re.compile(r"^H\d+$"),
        re.compile(r"^portfolio:[a-z0-9_]+$"),
    )
    _DIR_1X2_MAP: Dict[str, Tuple[int, str]] = {
        "主胜": (0, "H"),
        "胜": (0, "H"),
        "平": (1, "D"),
        "客胜": (2, "A"),
        "负": (2, "A"),
    }

    @staticmethod
    def resolve_ledger_path(ledger_path: Optional[str]) -> Tuple[Optional[pathlib.Path], Optional[dict]]:
        env_path = os.environ.get("SPORTS_MATH_BAYESIAN_LEDGER", "").strip()
        raw = (ledger_path or "").strip()
        if not raw:
            if not env_path:
                return None, {
                    "error_code": "INVALID_LEDGER_PATH",
                    "message": "ledger_path is required when SPORTS_MATH_BAYESIAN_LEDGER is unset. Pass an absolute path to data/bayesian_tracker.jsonl."
                }
            p = pathlib.Path(env_path)
            if not p.is_absolute():
                return None, {
                    "error_code": "INVALID_LEDGER_PATH",
                    "message": f"SPORTS_MATH_BAYESIAN_LEDGER must be an absolute path, got {env_path!r}."
                }
            return p, None
        p = pathlib.Path(raw)
        if not p.is_absolute():
            if env_path and pathlib.Path(env_path).is_absolute():
                return pathlib.Path(env_path), None
            return None, {
                "error_code": "INVALID_LEDGER_PATH",
                "received": raw,
                "message": f"ledger_path must be an absolute path (got relative {raw!r}) when SPORTS_MATH_BAYESIAN_LEDGER is unset."
            }
        return p, None

    @classmethod
    def normalize_entity_id(cls, raw: Any) -> Optional[str]:
        if not isinstance(raw, str):
            return None
        s = re.split(r"[#\s§]", raw.strip(), maxsplit=1)[0]
        s = s.lstrip("./")
        for prefix in ("skills/deep-analysis/wiki/", "deep-analysis/wiki/", "wiki/"):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        s = re.sub(r"^(playbooks|features)/\d{2}-", r"\1/", s)
        if re.match(r"^(playbooks|features)/[a-z0-9][a-z0-9-]*$", s):
            s = f"{s}.md"
        for pat in cls._ENTITY_PATTERNS:
            if pat.match(s):
                return s
        return None

    @classmethod
    def infer_entity_type(cls, entity_id: str, explicit_type: Optional[str] = None) -> str:
        if isinstance(explicit_type, str) and explicit_type.strip():
            et = explicit_type.strip().lower()
            alias = {"hypothesis": "draft", "stable": "playbook"}
            et = alias.get(et, et)
            if et in cls.VALID_ENTITY_TYPES:
                return et
        if entity_id.startswith("H"):
            return "draft"
        if entity_id.startswith("features/"):
            return "feature"
        if entity_id.startswith("portfolio:"):
            return "portfolio"
        return "playbook"

    @classmethod
    def clamp_baseline(cls, p_val: Any) -> Tuple[Optional[float], Optional[str]]:
        if not _is_strict_finite_number(p_val):
            return None, "INVALID_P_BASELINE"
        fv = float(p_val)
        if fv <= 0.0 or fv >= 1.0:
            return None, "INVALID_P_BASELINE"
        clamped = max(cls.P_MIN, min(cls.P_MAX, fv))
        return round(clamped, 6), None

    @classmethod
    def sort_key(cls, row: Dict[str, Any]) -> Tuple[str, str, str, str]:
        dt = ConformalEngine.parse_timestamp(row.get("kickoff_at"))
        ko_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else "9999-99-99T99:99:99Z"
        return (
            str(row.get("sales_date", "")),
            ko_iso,
            str(row.get("match_id", "")),
            str(row.get("entity_id", "")),
        )

    @classmethod
    def evaluate_entity(
        cls,
        entity_id: str,
        entity_type: str,
        rows: List[Dict[str, Any]],
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        decay_gamma: float = DEFAULT_DECAY_GAMMA,
        stale_after: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        sorted_rows = sorted(rows, key=cls.sort_key)
        n_obs = len(sorted_rows)
        stale_dt = ConformalEngine.parse_timestamp(stale_after) if stale_after else None
        as_of_dt = ConformalEngine.parse_timestamp(as_of) if as_of else None

        if n_obs == 0:
            if entity_type == "draft" and stale_dt is not None and as_of_dt is not None and as_of_dt >= stale_dt:
                status = "STALE_REVIEW"
            else:
                status = "COLLECTING" if entity_type == "draft" else "HEALTHY"
            return {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "n_obs": 0,
                "hits": 0,
                "mean_p_baseline": None,
                "posterior_mean": None,
                "ci90": None,
                "p_beat_market": None,
                "p_below_market": None,
                "e_pos": 1.0,
                "e_neg": 1.0,
                "anytime_p_pos": 1.0,
                "decayed_n_eff": 0.0,
                "decayed_posterior_mean": None,
                "decayed_p_below_market": None,
                "cusum_drift_e": 1.0,
                "advisory_status": status,
            }

        ys = [int(r["hit"]) for r in sorted_rows]
        p0s = [max(cls.P_MIN, min(cls.P_MAX, float(r["p_baseline"]))) for r in sorted_rows]
        hits = sum(ys)
        mean_p0 = sum(p0s) / n_obs

        m0 = max(0.5, float(prior_strength))
        a_full = m0 * mean_p0 + hits
        b_full = m0 * (1.0 - mean_p0) + (n_obs - hits)
        post_mean = a_full / (a_full + b_full)
        ci_lo = beta_ppf(0.05, a_full, b_full)
        ci_hi = beta_ppf(0.95, a_full, b_full)
        p_below = regularized_incomplete_beta(mean_p0, a_full, b_full)
        p_beat = 1.0 - p_below

        pos_wealths = [1.0] * len(cls.LAMBDA_GRID)
        neg_wealths = [1.0] * len(cls.LAMBDA_GRID)
        cusum_wealths = [1.0] * len(cls.LAMBDA_GRID)
        max_e_pos = 1.0

        for y_t, p_t in zip(ys, p0s):
            for idx, lam in enumerate(cls.LAMBDA_GRID):
                pos_factor = 1.0 + lam * (y_t - p_t) / p_t
                neg_factor = 1.0 - lam * (y_t - p_t) / (1.0 - p_t)
                pos_wealths[idx] *= pos_factor
                neg_wealths[idx] *= neg_factor
                cusum_wealths[idx] = max(1.0, cusum_wealths[idx] * neg_factor)
            e_pos_step = sum(pos_wealths) / len(cls.LAMBDA_GRID)
            if e_pos_step > max_e_pos:
                max_e_pos = e_pos_step

        e_pos = sum(pos_wealths) / len(cls.LAMBDA_GRID)
        e_neg = sum(neg_wealths) / len(cls.LAMBDA_GRID)
        cusum_drift_e = sum(cusum_wealths) / len(cls.LAMBDA_GRID)
        anytime_p_pos = min(1.0, 1.0 / max(1.0, max_e_pos))

        gamma = max(0.50, min(0.999, float(decay_gamma)))
        dec_n = 0.0
        dec_hits = 0.0
        dec_p0_sum = 0.0
        for idx, (y_t, p_t) in enumerate(zip(ys, p0s)):
            w = gamma ** (n_obs - 1 - idx)
            dec_n += w
            dec_hits += w * y_t
            dec_p0_sum += w * p_t
        dec_p0 = dec_p0_sum / dec_n
        a_dec = m0 * dec_p0 + dec_hits
        b_dec = m0 * (1.0 - dec_p0) + (dec_n - dec_hits)
        dec_post_mean = a_dec / (a_dec + b_dec)
        dec_p_below = regularized_incomplete_beta(dec_p0, a_dec, b_dec)

        if entity_type == "draft":
            if n_obs >= 5 and p_beat >= 0.90 and e_pos >= 20.0:
                status = "STRONG_EVIDENCE"
            elif n_obs >= 5 and p_beat >= 0.90 and e_pos >= 5.0:
                status = "PROMOTE_RECOMMENDED"
            elif n_obs >= 5 and p_below >= 0.85 and e_neg >= 5.0:
                status = "DEPRECATE_RECOMMENDED"
            elif stale_dt is not None and as_of_dt is not None and as_of_dt >= stale_dt:
                status = "STALE_REVIEW"
            else:
                status = "COLLECTING"
        else:
            if dec_n >= 8.0 and (dec_p_below >= 0.90 or cusum_drift_e >= 8.0):
                status = "DRIFT_ALERT"
            elif dec_n >= 5.0 and (dec_p_below >= 0.80 or cusum_drift_e >= 5.0):
                status = "DRIFT_WATCH"
            else:
                status = "HEALTHY"

        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "n_obs": n_obs,
            "hits": hits,
            "mean_p_baseline": round(mean_p0, 6),
            "posterior_mean": round(post_mean, 6),
            "ci90": [round(ci_lo, 6), round(ci_hi, 6)],
            "p_beat_market": round(p_beat, 6),
            "p_below_market": round(p_below, 6),
            "e_pos": round(e_pos, 4),
            "e_neg": round(e_neg, 4),
            "anytime_p_pos": round(anytime_p_pos, 6),
            "decayed_n_eff": round(dec_n, 4),
            "decayed_posterior_mean": round(dec_post_mean, 6),
            "decayed_p_below_market": round(dec_p_below, 6),
            "cusum_drift_e": round(cusum_drift_e, 4),
            "advisory_status": status,
        }

    @classmethod
    def record_outcomes(
        cls,
        records: Any,
        ledger_path: Optional[str] = None,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        decay_gamma: float = DEFAULT_DECAY_GAMMA,
    ) -> dict:
        if not isinstance(records, list) or len(records) == 0:
            return {
                "status": "error",
                "error": {"error_code": "EMPTY_RECORDS", "message": "records must be a non-empty list."}
            }

        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "error": l_err}
        assert lpath is not None

        existing, read_err = ConformalEngine.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "error": read_err}

        existing_keys = {
            (str(r.get("match_id", "")), str(r.get("entity_id", "")))
            for r in existing
        }
        accepted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []
        touched_entities = set()

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                skipped.append({"index": str(idx), "reason": "INVALID_RECORD_TYPE"})
                continue
            mid = str(rec.get("match_id", "")).strip()
            if not mid:
                skipped.append({"index": str(idx), "reason": "MISSING_MATCH_ID"})
                continue
            if _is_truthy_flag(rec.get("known_answer_replay")):
                skipped.append({"match_id": mid, "reason": "KNOWN_ANSWER_REPLAY_REJECTED"})
                continue
            norm_eid = cls.normalize_entity_id(rec.get("entity_id"))
            if norm_eid is None:
                skipped.append({"match_id": mid, "reason": "INVALID_ENTITY_ID"})
                continue
            if (mid, norm_eid) in existing_keys:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "DUPLICATE_OBSERVATION"})
                continue

            sd_str = str(rec.get("sales_date", "")).strip()
            if not sd_str or not re.match(r"^\d{4}-\d{2}-\d{2}$", sd_str):
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "INVALID_SALES_DATE"})
                continue
            ko_str = str(rec.get("kickoff_at", "")).strip()
            ko_dt = ConformalEngine.parse_timestamp(ko_str)
            if ko_dt is None:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "INVALID_KICKOFF_AT"})
                continue
            ct_str = str(rec.get("cutoff_time", "")).strip()
            ct_dt = ConformalEngine.parse_timestamp(ct_str)
            if ct_dt is None:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "MISSING_OR_INVALID_CUTOFF"})
                continue
            if ct_dt >= ko_dt:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "NON_PREMATCH_CUTOFF"})
                continue

            raw_hit = rec.get("hit")
            if isinstance(raw_hit, bool):
                hit_val = 1 if raw_hit else 0
            elif isinstance(raw_hit, int) and raw_hit in (0, 1):
                hit_val = int(raw_hit)
            else:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "INVALID_HIT"})
                continue

            clamped_p0, p_err = cls.clamp_baseline(rec.get("p_baseline"))
            if p_err is not None or clamped_p0 is None:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": p_err or "INVALID_P_BASELINE"})
                continue

            etype = cls.infer_entity_type(norm_eid, rec.get("entity_type"))
            row_obj: Dict[str, Any] = {
                "match_id": mid,
                "sales_date": sd_str,
                "kickoff_at": ko_dt.astimezone(ConformalEngine.BJ_TZ).isoformat(),
                "cutoff_time": ct_dt.astimezone(ConformalEngine.BJ_TZ).isoformat(),
                "entity_id": norm_eid,
                "entity_type": etype,
                "direction": str(rec.get("direction", "")).strip(),
                "p_baseline": clamped_p0,
                "hit": hit_val,
                "source_card": str(rec.get("source_card", "")).strip(),
            }
            existing_keys.add((mid, norm_eid))
            accepted.append(row_obj)
            touched_entities.add(norm_eid)

        combined = sorted(existing + accepted, key=cls.sort_key)
        if accepted:
            ConformalEngine.atomic_write_ledger(lpath, combined)

        by_eid: Dict[str, List[Dict[str, Any]]] = {}
        for r in combined:
            by_eid.setdefault(str(r["entity_id"]), []).append(r)

        updated_summaries = [
            cls.evaluate_entity(
                eid,
                cls.infer_entity_type(eid, by_eid[eid][-1].get("entity_type")),
                by_eid[eid],
                prior_strength=prior_strength,
                decay_gamma=decay_gamma,
            )
            for eid in sorted(touched_entities)
        ]

        return {
            "status": "ok",
            "ledger_path": str(lpath),
            "added_count": len(accepted),
            "skipped": skipped,
            "total_rows": len(combined),
            "updated_entities": updated_summaries,
        }

    @classmethod
    def health_report(
        cls,
        ledger_path: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        as_of: Optional[str] = None,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        decay_gamma: float = DEFAULT_DECAY_GAMMA,
    ) -> dict:
        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "error": l_err}
        assert lpath is not None

        rows, read_err = ConformalEngine.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "error": read_err}

        by_eid: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            norm = cls.normalize_entity_id(r.get("entity_id")) or str(r.get("entity_id", ""))
            if norm:
                by_eid.setdefault(norm, []).append(r)

        meta_norm: Dict[str, Dict[str, Any]] = {}
        if isinstance(entity_metadata, dict):
            for k, v in entity_metadata.items():
                norm_k = cls.normalize_entity_id(k) or str(k)
                if isinstance(v, dict):
                    meta_norm[norm_k] = v

        all_eids = set(by_eid.keys()) | set(meta_norm.keys())
        if entity_ids is not None:
            want = {cls.normalize_entity_id(x) or str(x) for x in entity_ids}
            all_eids = all_eids & want

        filter_type = entity_type.strip().lower() if isinstance(entity_type, str) and entity_type.strip() else None
        entities: List[Dict[str, Any]] = []
        for eid in sorted(all_eids):
            erows = by_eid.get(eid, [])
            m = meta_norm.get(eid, {})
            explicit_t = m.get("entity_type") or (erows[-1].get("entity_type") if erows else None)
            etype = cls.infer_entity_type(eid, explicit_t)
            if filter_type and etype != filter_type:
                continue
            entities.append(
                cls.evaluate_entity(
                    eid,
                    etype,
                    erows,
                    prior_strength=prior_strength,
                    decay_gamma=decay_gamma,
                    stale_after=m.get("stale_after"),
                    as_of=as_of,
                )
            )

        alerts = {
            "promote_candidates": [e["entity_id"] for e in entities if e["advisory_status"] in ("PROMOTE_RECOMMENDED", "STRONG_EVIDENCE")],
            "deprecate_candidates": [e["entity_id"] for e in entities if e["advisory_status"] == "DEPRECATE_RECOMMENDED"],
            "stale_reviews": [e["entity_id"] for e in entities if e["advisory_status"] == "STALE_REVIEW"],
            "drift_watches": [e["entity_id"] for e in entities if e["advisory_status"] == "DRIFT_WATCH"],
            "drift_alerts": [e["entity_id"] for e in entities if e["advisory_status"] == "DRIFT_ALERT"],
        }

        return {
            "status": "ok",
            "algorithm": "Beta-MixtureEProcess-PageCUSUM",
            "ledger_path": str(lpath),
            "total_rows": len(rows),
            "entity_count": len(entities),
            "entities": entities,
            "alerts": alerts,
        }

    @classmethod
    def verify_ledger_file(cls, ledger_path: str, min_rows: int = 50) -> Tuple[bool, str]:
        p = pathlib.Path(ledger_path)
        rows, err = ConformalEngine.load_ledger(p)
        if err is not None:
            return False, f"ERR {err}"
        if len(rows) < min_rows:
            return False, f"ERR row_count={len(rows)} < min_rows={min_rows}"
        seen_keys = set()
        prev_key: Optional[Tuple[str, str, str, str]] = None
        for idx, r in enumerate(rows):
            mid = str(r.get("match_id", ""))
            eid = str(r.get("entity_id", ""))
            if not mid or not eid or (mid, eid) in seen_keys:
                return False, f"ERR duplicate_or_empty_key at row {idx}: {(mid, eid)!r}"
            seen_keys.add((mid, eid))
            if cls.normalize_entity_id(eid) != eid:
                return False, f"ERR non_canonical_entity_id at {mid}: {eid!r}"
            if r.get("hit") not in (0, 1):
                return False, f"ERR invalid hit at {(mid, eid)}: {r.get('hit')!r}"
            pb = r.get("p_baseline")
            if not _is_strict_finite_number(pb) or float(pb) < cls.P_MIN - 1e-9 or float(pb) > cls.P_MAX + 1e-9:
                return False, f"ERR out_of_bounds p_baseline at {(mid, eid)}: {pb!r}"
            ko_dt = ConformalEngine.parse_timestamp(r.get("kickoff_at"))
            ct_dt = ConformalEngine.parse_timestamp(r.get("cutoff_time"))
            if ko_dt is None or ct_dt is None or ct_dt >= ko_dt:
                return False, f"ERR invalid temporal gate at {(mid, eid)}"
            key = cls.sort_key(r)
            if prev_key is not None and key < prev_key:
                return False, f"ERR unsorted ledger at {(mid, eid)}: {key} < {prev_key}"
            prev_key = key
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return True, f"BAYES_LEDGER_OK {len(rows)} sha256={digest}"

    @classmethod
    def bootstrap_from_repo(cls, repo_root: str, output_path: str) -> Tuple[int, str]:
        """Deterministically build Bayesian observation JSONL from settled prematch cards (pure 1X2 + portfolio)."""
        rows: List[Dict[str, Any]] = []
        seen_keys = set()
        pick_map = {"胜": (0, "H", "主胜"), "平": (1, "D", "平"), "负": (2, "A", "客胜")}

        for mid, sales_date, ko_dt, oc_dt, _ident, dec, ms, norm_p, outcome in ConformalEngine._iter_settled_prematch_cards(repo_root):
            ko_iso = ko_dt.astimezone(ConformalEngine.BJ_TZ).isoformat()
            ct_iso = oc_dt.astimezone(ConformalEngine.BJ_TZ).isoformat()
            src_card = f"data/{sales_date}/cards/{mid}.yaml"

            primary = str(dec.get("primary", "")).strip()
            if primary in pick_map and (mid, "portfolio:1x2_primary") not in seen_keys:
                p_idx, p_cls, p_dir = pick_map[primary]
                p0, _ = cls.clamp_baseline(norm_p[p_idx])
                assert p0 is not None
                seen_keys.add((mid, "portfolio:1x2_primary"))
                rows.append({
                    "match_id": mid,
                    "sales_date": sales_date,
                    "kickoff_at": ko_iso,
                    "cutoff_time": ct_iso,
                    "entity_id": "portfolio:1x2_primary",
                    "entity_type": "portfolio",
                    "direction": p_dir,
                    "p_baseline": p0,
                    "hit": 1 if outcome == p_cls else 0,
                    "source_card": src_card,
                })

            for h in (ms.get("wiki_hits") or []):
                if not isinstance(h, dict):
                    continue
                eid = cls.normalize_entity_id(h.get("page"))
                if not eid or not eid.startswith("playbooks/"):
                    continue
                direction = str(h.get("direction", "")).strip()
                if direction not in cls._DIR_1X2_MAP:
                    continue
                if (mid, eid) in seen_keys:
                    continue
                d_idx, d_cls = cls._DIR_1X2_MAP[direction]
                p0, _ = cls.clamp_baseline(norm_p[d_idx])
                assert p0 is not None
                seen_keys.add((mid, eid))
                rows.append({
                    "match_id": mid,
                    "sales_date": sales_date,
                    "kickoff_at": ko_iso,
                    "cutoff_time": ct_iso,
                    "entity_id": eid,
                    "entity_type": "playbook",
                    "direction": direction,
                    "p_baseline": p0,
                    "hit": 1 if outcome == d_cls else 0,
                    "source_card": src_card,
                })

        rows.sort(key=cls.sort_key)
        out_p = pathlib.Path(output_path)
        ConformalEngine.atomic_write_ledger(out_p, rows)
        ok, msg = cls.verify_ledger_file(str(out_p), min_rows=0)
        if not ok:
            raise RuntimeError(f"Bayesian bootstrap failed verification: {msg}")
        return len(rows), msg


@mcp.tool()
def devig_oo_epc(home_odds: float, draw_odds: float, away_odds: float) -> dict:
    """Calculate true unbiased 1X2 probabilities from bookmaker odds using Tokyo University Goto (2026) OO-EPC algorithm.

    Args:
        home_odds: Bookmaker decimal odds for Home Win (e.g. 1.51)
        draw_odds: Bookmaker decimal odds for Draw (e.g. 4.30)
        away_odds: Bookmaker decimal odds for Away Win (e.g. 4.60)
    """
    parsed, err = OoEpcCalculator.validate_triple([home_odds, draw_odds, away_odds], context="devig_oo_epc")
    if err is not None:
        return {"status": "error", "algorithm": "Goto-OO-EPC", "error": err}
    return OoEpcCalculator.solve_triple(parsed)


@mcp.tool()
def devig_oo_epc_batch(items: List[Dict[str, Any]]) -> dict:
    """Calculate Goto (2026) OO-EPC de-vigged probabilities, bookmaker margins, opening-to-latest shifts, and Pinnacle (cid=177) scissor gaps for up to 21 bookmakers in one call.

    Args:
        items: List of bookmaker dicts. Each dict accepts:
            - cid: Bookmaker integer/string ID (e.g. 177 for Pinnacle, 1129 for Sporttery, 545 for Crown, 281 for Bet365)
            - name / bookmaker: Optional bookmaker display name (e.g. "平博", "中国体彩")
            - open / open_odds: Optional 3-element list [home_odds, draw_odds, away_odds] for opening 1X2 odds
            - latest / latest_odds / odds: Optional 3-element list [home_odds, draw_odds, away_odds] for latest 1X2 odds
    """
    if not isinstance(items, list) or len(items) == 0:
        return {
            "status": "error",
            "algorithm": "Goto-OO-EPC",
            "error": {
                "error_code": "EMPTY_ITEMS",
                "message": "items must be a non-empty list of bookmaker dicts, e.g. [{'cid': 177, 'name': '平博', 'open': [1.41, 4.91, 7.60], 'latest': [1.43, 4.86, 6.72]}]."
            }
        }

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    pinnacle_anchor: Optional[Dict[str, Any]] = None

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append({
                "index": idx,
                "error_code": "INVALID_ITEM_TYPE",
                "message": f"items[{idx}] must be a dict, got {type(item).__name__}."
            })
            continue

        cid = item.get("cid", item.get("机构编号", idx))
        name = item.get("name", item.get("bookmaker", item.get("机构", str(cid))))
        label = f"cid={cid}({name})"

        raw_open = item.get("open", item.get("open_odds"))
        raw_latest = item.get("latest", item.get("latest_odds", item.get("odds")))

        if raw_open is None and raw_latest is None:
            errors.append({
                "cid": cid,
                "name": name,
                "error_code": "MISSING_ODDS",
                "message": f"{label} must provide 'open' ([h,d,a]) and/or 'latest' ([h,d,a])."
            })
            continue

        row_out: Dict[str, Any] = {"cid": cid, "name": name}
        open_res: Optional[dict] = None
        latest_res: Optional[dict] = None

        if raw_open is not None:
            parsed_open, err_open = OoEpcCalculator.validate_triple(raw_open, context=f"{label}.open")
            if err_open is not None:
                errors.append({"cid": cid, "name": name, **err_open})
                continue
            open_res = OoEpcCalculator.solve_triple(parsed_open)
            row_out["open"] = {
                "odds": open_res["odds"],
                "prob_1x2": open_res["prob_1x2"],
                "percentages_1x2": open_res["percentages_1x2"],
                "display_text": open_res["display_text"],
                "margin_pct": open_res["margin_pct"]
            }

        if raw_latest is not None:
            parsed_latest, err_latest = OoEpcCalculator.validate_triple(raw_latest, context=f"{label}.latest")
            if err_latest is not None:
                errors.append({"cid": cid, "name": name, **err_latest})
                continue
            latest_res = OoEpcCalculator.solve_triple(parsed_latest)
            row_out["latest"] = {
                "odds": latest_res["odds"],
                "prob_1x2": latest_res["prob_1x2"],
                "percentages_1x2": latest_res["percentages_1x2"],
                "display_text": latest_res["display_text"],
                "margin_pct": latest_res["margin_pct"]
            }

        if open_res is not None and latest_res is not None:
            dh = round((latest_res["prob_1x2"]["home"] - open_res["prob_1x2"]["home"]) * 100.0, 2)
            dd = round((latest_res["prob_1x2"]["draw"] - open_res["prob_1x2"]["draw"]) * 100.0, 2)
            da = round((latest_res["prob_1x2"]["away"] - open_res["prob_1x2"]["away"]) * 100.0, 2)
            dm = round(latest_res["margin_pct"] - open_res["margin_pct"], 2)
            row_out["delta_prob_pct"] = {"home": dh, "draw": dd, "away": da}
            row_out["delta_margin_pct"] = dm

        is_pinnacle = (str(cid) == "177") or ("平博" in str(name)) or ("pinnacle" in str(name).lower())
        if is_pinnacle and latest_res is not None:
            pinnacle_anchor = {
                "cid": cid,
                "name": name,
                "latest_prob_1x2": latest_res["prob_1x2"],
                "latest_display_text": latest_res["display_text"],
                "latest_margin_pct": latest_res["margin_pct"],
                "delta_prob_pct": row_out.get("delta_prob_pct")
            }

        results.append(row_out)

    if pinnacle_anchor is not None:
        pin_p = pinnacle_anchor["latest_prob_1x2"]
        for r in results:
            if "latest" in r:
                lp = r["latest"]["prob_1x2"]
                r["gap_vs_pin_latest_pct"] = {
                    "home": round((lp["home"] - pin_p["home"]) * 100.0, 2),
                    "draw": round((lp["draw"] - pin_p["draw"]) * 100.0, 2),
                    "away": round((lp["away"] - pin_p["away"]) * 100.0, 2)
                }

    return {
        "status": "success" if len(errors) == 0 else ("partial" if len(results) > 0 else "error"),
        "algorithm": "Goto-OO-EPC-Batch",
        "count": len(results),
        "pinnacle_anchor": pinnacle_anchor,
        "results": results,
        "errors": errors
    }


@mcp.tool()
def poisson_dixon_coles(home_lambda: float, away_lambda: float, max_goals: int = 8, top_n: int = 6) -> dict:
    """Calculate bivariate Poisson match score probabilities with Dixon-Coles low-score adjustments.

    Args:
        home_lambda: Expected goals for Home Team (e.g. 1.85)
        away_lambda: Expected goals for Away Team (e.g. 0.95)
        max_goals: Maximum goals per team considered (default: 8)
        top_n: Number of most probable scorelines to return (default: 6)
    """
    return PoissonEngine.calculate(home_lambda, away_lambda, max_goals=max_goals, top_n=top_n)


@mcp.tool()
def conformal_predict_1x2(
    probs: List[float],
    cutoff_time: Optional[str] = None,
    cutoff_sales_date: Optional[str] = None,
    alpha: float = 0.20,
    window_size: int = 150,
    ledger_path: Optional[str] = None,
) -> dict:
    """Compute causal 1X2 conformal prediction set (LAC), Mondrian draw alert, ACI alpha_t, and Inductive Venn-Abers intervals.

    Args:
        probs: 3-element list [p_home, p_draw, p_away] (either 0..1 or 0..100 percentages)
        cutoff_time: ISO-8601 pre-match cutoff timestamp with timezone (enforces kickoff_at <= cutoff_time - 3h)
        cutoff_sales_date: Optional YYYY-MM-DD fallback cutoff (used if cutoff_time is omitted)
        alpha: Nominal target miscoverage rate (default 0.20 for 80% coverage)
        window_size: Rolling window cap of most recent causal calibration matches (default 150)
        ledger_path: Absolute path to data/conformal_calibration.jsonl (falls back to SPORTS_MATH_CONFORMAL_LEDGER)
    """
    return ConformalEngine.predict_1x2(
        probs=probs,
        cutoff_time=cutoff_time,
        cutoff_sales_date=cutoff_sales_date,
        alpha=alpha,
        window_size=window_size,
        ledger_path=ledger_path,
    )


@mcp.tool()
def conformal_record_outcomes(
    records: List[Dict[str, Any]],
    ledger_path: Optional[str] = None,
    alpha_target: float = 0.20,
    gamma: float = 0.05,
) -> dict:
    """Idempotently append settled pre-match 1X2 outcomes to conformal_calibration.jsonl and update ACI alpha_t.

    Args:
        records: List of settled match dicts (match_id, sales_date, kickoff_at, league, probs, outcome, source_card, cutoff_time, known_answer_replay, optional prediction_set, alpha_t)
        ledger_path: Absolute path to data/conformal_calibration.jsonl
        alpha_target: Target miscoverage rate (default 0.20)
        gamma: ACI step size (default 0.05; alpha_t clipped to [0.10, 0.40])
    """
    return ConformalEngine.record_outcomes(
        records=records,
        ledger_path=ledger_path,
        alpha_target=alpha_target,
        gamma=gamma,
    )


@mcp.tool()
def bayesian_record_outcomes(
    records: List[Dict[str, Any]],
    ledger_path: Optional[str] = None,
    prior_strength: float = 4.0,
    decay_gamma: float = 0.95,
) -> dict:
    """Idempotently append settled pre-match directional/portfolio observations at (match_id, entity_id) grain.

    Args:
        records: List of settled observation dicts (match_id, sales_date, kickoff_at, cutoff_time, entity_id, entity_type, direction, p_baseline, hit, source_card, known_answer_replay)
        ledger_path: Absolute path to data/bayesian_tracker.jsonl (falls back to SPORTS_MATH_BAYESIAN_LEDGER)
        prior_strength: Pseudo-count prior M_0 anchored at market mean baseline (default 4.0)
        decay_gamma: Exponential forgetting factor for drift detection (default 0.95)
    """
    return BayesianTrackerEngine.record_outcomes(
        records=records,
        ledger_path=ledger_path,
        prior_strength=prior_strength,
        decay_gamma=decay_gamma,
    )


@mcp.tool()
def bayesian_health_report(
    ledger_path: Optional[str] = None,
    entity_ids: Optional[List[str]] = None,
    entity_type: Optional[str] = None,
    entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    as_of: Optional[str] = None,
    prior_strength: float = 4.0,
    decay_gamma: float = 0.95,
) -> dict:
    """Evaluate market-anchored Beta posteriors, anytime-valid e-processes, discounted posteriors, and Page-CUSUM drift alerts.

    Args:
        ledger_path: Absolute path to data/bayesian_tracker.jsonl (falls back to SPORTS_MATH_BAYESIAN_LEDGER)
        entity_ids: Optional filter list of entity IDs
        entity_type: Optional filter ('draft', 'playbook', 'feature', 'portfolio')
        entity_metadata: Optional map {entity_id: {entity_type, stale_after}} enabling STALE_REVIEW for 0-observation drafts
        as_of: Optional ISO-8601 timestamp for evaluating stale_after deadlines
        prior_strength: Pseudo-count prior M_0 anchored at market mean baseline (default 4.0)
        decay_gamma: Exponential forgetting factor for drift detection (default 0.95)
    """
    return BayesianTrackerEngine.health_report(
        ledger_path=ledger_path,
        entity_ids=entity_ids,
        entity_type=entity_type,
        entity_metadata=entity_metadata,
        as_of=as_of,
        prior_strength=prior_strength,
        decay_gamma=decay_gamma,
    )


def _cli_main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--bootstrap-ledger", action="store_true")
    parser.add_argument("--bootstrap-bayesian-ledger", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="data/conformal_calibration.jsonl")
    parser.add_argument("--verify-ledger", default=None)
    parser.add_argument("--verify-bayesian-ledger", default=None)
    parser.add_argument("--min-rows", type=int, default=50)
    args, _ = parser.parse_known_args()

    if args.bootstrap_ledger:
        _, msg = ConformalEngine.bootstrap_from_repo(args.repo_root, args.output)
        print(msg)
        return 0
    if args.bootstrap_bayesian_ledger:
        out_path = args.output if args.output != "data/conformal_calibration.jsonl" else "data/bayesian_tracker.jsonl"
        _, msg = BayesianTrackerEngine.bootstrap_from_repo(args.repo_root, out_path)
        print(msg)
        return 0
    if args.verify_ledger:
        ok, msg = ConformalEngine.verify_ledger_file(args.verify_ledger, min_rows=args.min_rows)
        print(msg)
        return 0 if ok else 1
    if args.verify_bayesian_ledger:
        ok, msg = BayesianTrackerEngine.verify_ledger_file(args.verify_bayesian_ledger, min_rows=args.min_rows)
        print(msg)
        return 0 if ok else 1
    mcp.run()
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
