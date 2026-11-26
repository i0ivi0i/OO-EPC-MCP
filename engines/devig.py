"""Pure-stdlib OO-EPC odds devigging and Dixon-Coles bivariate Poisson engines."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class OoEpcCalculator:
    """Tokyo University Shota Goto (2026) OO-EPC De-vigging Algorithm."""

    @staticmethod
    def validate_triple(odds: List[Any], context: str = "odds") -> Tuple[Optional[List[float]], Optional[dict]]:
        if not isinstance(odds, (list, tuple)) or len(odds) != 3:
            return None, {
                "error_code": "INVALID_TRIPLE_LENGTH",
                "field": context,
                "received": odds,
                "message": f"{context} must be a 3-element list [home_odds, draw_odds, away_odds], got {odds!r}. Retry with exactly 3 decimal odds >= 1.0.",
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
                    "message": f"{context}.{labels[idx]} must be a numeric decimal odd >= 1.0, got {val!r}.",
                }
            if not math.isfinite(fval) or fval < 1.0:
                return None, {
                    "error_code": "ODD_OUT_OF_RANGE",
                    "field": f"{context}.{labels[idx]}",
                    "received": fval,
                    "message": f"{context}.{labels[idx]} must be >= 1.0 (received {fval}). Check column order (do not pass return-rate or handicap line as 1X2 odds).",
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
                "away": round(probs[2], 6),
            },
            "percentages_1x2": {
                "home": f"{h_pct:.2f}%",
                "draw": f"{d_pct:.2f}%",
                "away": f"{a_pct:.2f}%",
            },
        }

    @classmethod
    def solve_batch(cls, items: Any) -> dict:
        """Batch OO-EPC de-vigging across bookmakers with opening/latest shift & Pinnacle anchor comparison."""
        if not isinstance(items, list) or len(items) == 0:
            return {
                "status": "error",
                "algorithm": "Goto-OO-EPC",
                "error": {
                    "error_code": "EMPTY_ITEMS",
                    "message": "items must be a non-empty list of bookmaker dicts, e.g. [{'cid': 177, 'name': '平博', 'open': [1.41, 4.91, 7.60], 'latest': [1.43, 4.86, 6.72]}].",
                },
            }

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        pinnacle_anchor: Optional[Dict[str, Any]] = None

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append({"index": idx, "error_code": "INVALID_ITEM_TYPE", "message": f"items[{idx}] must be a dict, got {type(item).__name__}."})
                continue
            cid = item.get("cid", item.get("机构编号", idx))
            name = item.get("name", item.get("bookmaker", item.get("机构", str(cid))))
            label = f"cid={cid}({name})"
            raw_open = item.get("open", item.get("open_odds"))
            raw_latest = item.get("latest", item.get("latest_odds", item.get("odds")))
            if raw_open is None and raw_latest is None:
                errors.append({"cid": cid, "name": name, "error_code": "MISSING_ODDS", "message": f"{label} must provide 'open' ([h,d,a]) and/or 'latest' ([h,d,a])."})
                continue

            row_out: Dict[str, Any] = {"cid": cid, "name": name}
            solved_phases: Dict[str, dict] = {}
            phase_failed = False
            for phase_key, raw_triple in (("open", raw_open), ("latest", raw_latest)):
                if raw_triple is None:
                    continue
                parsed_triple, err_triple = cls.validate_triple(raw_triple, context=f"{label}.{phase_key}")
                if err_triple is not None:
                    errors.append({"cid": cid, "name": name, **err_triple})
                    phase_failed = True
                    break
                assert parsed_triple is not None
                solved = cls.solve_triple(parsed_triple)
                solved_phases[phase_key] = solved
                row_out[phase_key] = {k: solved[k] for k in ("odds", "prob_1x2", "percentages_1x2", "display_text", "margin_pct")}
            if phase_failed:
                continue

            open_res = solved_phases.get("open")
            latest_res = solved_phases.get("latest")

            if open_res is not None and latest_res is not None:
                row_out["delta_prob_pct"] = {k: round((latest_res["prob_1x2"][k] - open_res["prob_1x2"][k]) * 100.0, 2) for k in ("home", "draw", "away")}
                row_out["delta_margin_pct"] = round(latest_res["margin_pct"] - open_res["margin_pct"], 2)

            if latest_res is not None and ((str(cid) == "177") or ("平博" in str(name)) or ("pinnacle" in str(name).lower())):
                pinnacle_anchor = {
                    "cid": cid,
                    "name": name,
                    "latest_prob_1x2": latest_res["prob_1x2"],
                    "latest_display_text": latest_res["display_text"],
                    "latest_margin_pct": latest_res["margin_pct"],
                    "delta_prob_pct": row_out.get("delta_prob_pct"),
                }
            results.append(row_out)

        if pinnacle_anchor is not None:
            pin_p = pinnacle_anchor["latest_prob_1x2"]
            for r in results:
                if "latest" in r:
                    lp = r["latest"]["prob_1x2"]
                    r["gap_vs_pin_latest_pct"] = {k: round((lp[k] - pin_p[k]) * 100.0, 2) for k in ("home", "draw", "away")}

        return {
            "status": "success" if len(errors) == 0 else ("partial" if len(results) > 0 else "error"),
            "algorithm": "Goto-OO-EPC-Batch",
            "count": len(results),
            "pinnacle_anchor": pinnacle_anchor,
            "results": results,
            "errors": errors,
        }


class PoissonEngine:
    """Bivariate Poisson with Dixon-Coles Low-Score Factor Adjustments."""

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

        pmf1 = [cls._pmf(l1, i) for i in range(max_goals)]
        pmf2 = [cls._pmf(l2, j) for j in range(max_goals)]
        raw_scores: List[Tuple[int, int, float]] = []
        for i in range(max_goals):
            p1 = pmf1[i]
            for j in range(max_goals):
                p = p1 * pmf2[j] * cls.DEFAULT_LOW_FACTORS.get((i, j), 1.0)
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
                "fair_odds": round(fair, 2),
            })

        return {
            "home_lambda": round(l1, 3),
            "away_lambda": round(l2, 3),
            "expected_total_goals": round(l1 + l2, 2),
            "distribution_1x2": {
                "home_win": f"{home_win_prob * 100:.2f}%",
                "draw": f"{draw_prob * 100:.2f}%",
                "away_win": f"{away_win_prob * 100:.2f}%",
            },
            "top_scores": top_list,
        }
