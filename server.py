"""
sports-math / OO-EPC MCP Server
Official Model Context Protocol server for sports betting analytics & odds mathematics.

Exposes:
- devig_oo_epc: Tokyo University Goto (2026) OO-EPC de-vigging convex optimization algorithm
- poisson_dixon_coles: Bivariate Poisson score matrix with low-score adjustment
"""

import math
from typing import Dict, List, Optional, Tuple

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
    version="1.0.0"
)


class OoEpcCalculator:
    """Tokyo University Shota Goto (2026) OO-EPC De-vigging Algorithm"""

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

        if any(p <= 0.0 for p in output_probs) or (booksum <= total):
            normalizer = booksum / total
            output_probs = [p / normalizer for p in inverses]

        return output_probs


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
    odds = [float(home_odds), float(draw_odds), float(away_odds)]
    probs = OoEpcCalculator.calculate(odds)
    h_pct = probs[0] * 100
    d_pct = probs[1] * 100
    a_pct = probs[2] * 100
    booksum = sum(1.0 / o for o in odds)
    margin = booksum - 1.0

    return {
        "algorithm": "Goto-OO-EPC",
        "odds": odds,
        "probabilities": [round(p, 6) for p in probs],
        "percentages": [f"{h_pct:.2f}%", f"{d_pct:.2f}%", f"{a_pct:.2f}%"],
        "display_text": f"{h_pct:.1f}% / {d_pct:.1f}% / {a_pct:.1f}%",
        "margin": round(margin, 4),
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
