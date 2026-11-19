"""
sports-math MCP Server
Official Model Context Protocol server for sports analytics & betting mathematics.
Exposes:
- devig_oo_epc: Tokyo University Goto (2026) OO-EPC de-vigging convex optimization
- devig_oo_epc_batch: Single-call batch OO-EPC de-vigging across up to 21 bookmakers with Pinnacle scissor-gap comparison
- poisson_dixon_coles: Bivariate Poisson score matrix with low-score adjustment
"""

import math
from typing import Any, Dict, List, Optional, Tuple

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
    description="Mathematical computation engine for sports betting odds devigging and score probability modeling.",
    version="1.1.1"
)


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


if __name__ == "__main__":
    mcp.run()
