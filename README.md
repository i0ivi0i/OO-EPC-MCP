# OO-EPC-MCP 📐⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/Protocol-MCP-blue.svg)](https://modelcontextprotocol.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![FastMCP](https://img.shields.io/badge/Framework-FastMCP-orange.svg)](https://github.com/PrefectHQ/fastmcp)

A lightweight, high-performance **Model Context Protocol (MCP)** server for sports betting analytics and actuarial mathematics. 

Built on **FastMCP** / official MCP Python SDK (`MCPServer`). Exposes pure mathematical algorithms with **zero external API calls**, **zero secret keys**, and **sub-millisecond local deterministic execution**.

---

## 🛠️ Tools Provided

### 1. `devig_oo_epc`
Calculates true unbiased 1X2 probabilities from 3-way bookmaker odds using the **Tokyo University Shota Goto (2026) OO-EPC (Odds-on Equal Profit Margin)** convex optimization algorithm.

- **Parameters**:
  - `home_odds` (`float`): Decimal odds for Home Win (e.g. `1.51`)
  - `draw_odds` (`float`): Decimal odds for Draw (e.g. `4.30`)
  - `away_odds` (`float`): Decimal odds for Away Win (e.g. `4.60`)
- **Returns**:
  - Unbiased true probability vector (`probabilities`, `percentages`)
  - Bookmaker overround / margin (`margin`)
  - Formatted display text (`63.4% / 19.1% / 17.5%`)

### 2. `poisson_dixon_coles`
Calculates bivariate Poisson match score probabilities with **Dixon-Coles** low-score factor adjustments (handling the systematic underestimation of low-scoring draws like 0-0, 1-1).

- **Parameters**:
  - `home_lambda` (`float`): Expected goals for Home Team (e.g. `1.85`)
  - `away_lambda` (`float`): Expected goals for Away Team (e.g. `0.95`)
  - `max_goals` (`int`, optional, default `8`): Upper bound of goals evaluated per side
  - `top_n` (`int`, optional, default `6`): Top $N$ most probable scorelines
- **Returns**:
  - Expected total goals
  - True 1X2 win/draw/loss probability distribution
  - Top scorelines ranked by probability with exact percentages and fair odds

---

## 🚀 Quick Start & Installation

### Option 1: Run with `uv` (Recommended)
No manual environment setup needed:
```bash
uv run server.py
```

### Option 2: Run with standard `pip`
```bash
pip install -r requirements.txt
python server.py
```

---

## 🔌 Client Integration

### Claude Desktop
Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sports-math": {
      "command": "python",
      "args": [
        "/path/to/OO-EPC-MCP/server.py"
      ]
    }
  }
}
```

### Hermes Agent
Add this to your `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  sports-math:
    command: python
    args:
      - /path/to/OO-EPC-MCP/server.py
```

### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "sports-math": {
      "command": "python",
      "args": ["/path/to/OO-EPC-MCP/server.py"]
    }
  }
}
```

---

## 📚 Mathematical Citations

- **Goto, Shota (2026)**. *Odds-on Equal Profit Margin (OO-EPC) Method for Multi-way Betting Market De-vigging*. The University of Tokyo.
- **Dixon, M. J., & Coles, S. G. (1997)**. *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Applied Statistics (Journal of the Royal Statistical Society: Series C), 46(2), 265–280.

---

## 📄 License

MIT License © 2026 [i0ivi0i](https://github.com/i0ivi0i)
