"""
Summarize an optimization result CSV (output of main.py) and optionally compare it with another.

Usage:
  python scripts/summarize_run.py RESULT.csv [--out summary.json]
  python scripts/summarize_run.py BASELINE.csv --compare NEW.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def monthly_returns(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    bench = df[df["ticker"] == "LQ45"].set_index("period")["pnl_percentage"] / 100
    port_rows = df[df["ticker"] != "LQ45"]
    port = (port_rows["allocations"] * port_rows["pnl_percentage"] / 100).groupby(port_rows["period"]).sum()
    months = pd.DataFrame({"portfolio": port, "lq45": bench}).dropna().sort_index()
    return months, port_rows


def stats(r: pd.Series) -> dict:
    growth = (1 + r).cumprod()
    years = len(r) / 12
    drawdown = growth / growth.cummax() - 1
    return {
        "total_return_pct": round((growth.iloc[-1] - 1) * 100, 2),
        "cagr_pct": round((growth.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "ann_vol_pct": round(r.std(ddof=1) * np.sqrt(12) * 100, 2),
        "sharpe_rf0": round(r.mean() / r.std(ddof=1) * np.sqrt(12), 2),
        "max_drawdown_pct": round(drawdown.min() * 100, 2),
        "best_month_pct": round(r.max() * 100, 2),
        "worst_month_pct": round(r.min() * 100, 2),
    }


def summarize(path: Path, cost: float = 0.002) -> dict:
    months, port_rows = monthly_returns(path)
    weights = port_rows.pivot_table(index="period", columns="ticker", values="allocations", fill_value=0)
    weights = weights.loc[weights.index.isin(months.index)]
    cash_months = int((weights.get("CASH", pd.Series(0, index=weights.index)) > 0).sum())
    weights = weights.drop(columns="CASH", errors="ignore")
    weights = weights.loc[weights.sum(axis=1) > 0]
    turnover = weights.diff().abs().sum(axis=1).iloc[1:] / 2
    holdings = (weights > 0).sum(axis=1)
    top = (weights.mean().sort_values(ascending=False).head(8) * 100).round(1)

    # Forecast quality: std of (monthly return / believed monthly vol) is ~1 when risk is forecast right
    risk_forecast = {}
    if "ex_ante_volatility" in port_rows.columns:
        believed = port_rows.groupby("period")["ex_ante_volatility"].first()
        believed = believed[(believed > 0) & believed.index.isin(months.index)]
        z = months.loc[believed.index, "portfolio"] / (believed / np.sqrt(12))
        risk_forecast = {
            "avg_believed_vol_pct": round(believed.mean() * 100, 2),
            "bias_statistic": round(z.std(ddof=1), 2),
            "avg_shrinkage_intensity": round(port_rows.groupby("period")["shrinkage_intensity"].first().mean(), 2),
        }

    # Trading costs: `cost` per unit of weight bought or sold, counting the first
    # month's purchase from cash. Weight drift within a month is ignored.
    held = port_rows[port_rows["ticker"] != "CASH"].pivot_table(
        index="period", columns="ticker", values="allocations", fill_value=0
    ).reindex(months.index, fill_value=0)
    traded = held.diff().abs().sum(axis=1)
    traded.iloc[0] = held.iloc[0].sum()
    net = stats(months["portfolio"] - cost * traded)
    net_of_costs = {
        "one_way_cost_pct": round(cost * 100, 2),
        "avg_monthly_cost_pct": round(cost * traded.mean() * 100, 3),
        "cagr_pct": net["cagr_pct"],
        "sharpe_rf0": net["sharpe_rf0"],
    }

    return {
        "file": str(path),
        "periods": f"{months.index.min()} .. {months.index.max()} ({len(months)} months)",
        "target_volatility_pct": round(float(port_rows["risk"].iloc[0]) * 100, 1),
        "portfolio": stats(months["portfolio"]),
        "lq45": stats(months["lq45"]),
        "months_beating_lq45_pct": round((months["portfolio"] > months["lq45"]).mean() * 100, 1),
        "cash_months": cash_months,
        "avg_holdings": round(holdings.mean(), 1),
        "avg_monthly_turnover_pct": round(turnover.mean() * 100, 1),
        "max_single_weight_pct": round(weights.max().max() * 100, 1),
        "top_avg_weights_pct": top.to_dict(),
        "risk_forecast": risk_forecast,
        "net_of_costs": net_of_costs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cost", type=float, default=0.002, help="one-way trading cost per unit traded (default 0.002 = 0.2%%)")
    args = parser.parse_args()

    base = summarize(args.result, args.cost)
    if args.compare:
        new = summarize(args.compare, args.cost)
        rows = []
        for side in ["portfolio", "lq45"]:
            for k in base[side]:
                rows.append({"metric": f"{side}.{k}", "before": base[side][k], "after": new[side][k]})
        for k in ["months_beating_lq45_pct", "cash_months", "avg_holdings", "avg_monthly_turnover_pct", "max_single_weight_pct"]:
            rows.append({"metric": k, "before": base[k], "after": new[k]})
        for k in base["net_of_costs"]:
            rows.append({"metric": f"net_of_costs.{k}", "before": base["net_of_costs"][k], "after": new["net_of_costs"][k]})
        for k in sorted(set(base["risk_forecast"]) | set(new["risk_forecast"])):
            rows.append({"metric": f"risk_forecast.{k}",
                         "before": base["risk_forecast"].get(k), "after": new["risk_forecast"].get(k)})
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nBefore top weights:", base["top_avg_weights_pct"])
        print("After top weights: ", new["top_avg_weights_pct"])
        result = {"before": base, "after": new}
    else:
        print(json.dumps(base, indent=2))
        result = base
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
