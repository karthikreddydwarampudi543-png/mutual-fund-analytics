import os
import sys
import pandas as pd


def _get_data_dir() -> str:
    if os.path.exists("data/processed/01_fund_master.csv"):
        return "data/processed"
    if os.path.exists("mutual-fund-analytics/data/processed/01_fund_master.csv"):
        return "mutual-fund-analytics/data/processed"
    raise FileNotFoundError("Could not locate data/processed/01_fund_master.csv")


def _infer_risk_tier_from_category(cat: str) -> str:
    c = str(cat).strip().lower()
    if any(k in c for k in ["liquid", "overnight", "money market", "debt", "bond", "gilt"]):
        return "Low"
    if any(k in c for k in ["hybrid", "balanced", "arbitrage", "conservative", "index"]):
        return "Moderate"
    return "High"


def _resolve_risk_column(df: pd.DataFrame) -> str | None:
    candidates = ["risk_grade", "risk_level", "risk_o_meter", "risk_ometer", "risk_category", "risk"]
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def recommend_funds(risk_appetite: str, top_n: int = 3) -> pd.DataFrame:
    """
    Recommend top N mutual fund schemes by Sharpe ratio matching the user's risk appetite.
    Supported risk_appetite inputs: 'Low', 'Moderate', 'High' (case-insensitive).
    """
    valid_tiers = {
        "low": ["low", "low to moderate", "low-moderate"],
        "moderate": ["moderate", "moderately high", "moderate-high", "medium"],
        "high": ["high", "very high", "very-high"],
    }
    key = risk_appetite.strip().lower()
    if key not in valid_tiers:
        raise ValueError(f"Invalid risk_appetite '{risk_appetite}'. Choose from: Low, Moderate, High.")

    data_dir = _get_data_dir()
    df_fund = pd.read_csv(f"{data_dir}/01_fund_master.csv")
    df_perf = pd.read_csv(f"{data_dir}/07_scheme_performance.csv")

    fund_cols = [
        c for c in ["amfi_code", "scheme_name", "fund_house", "category", "sub_category", "plan", "expense_ratio_pct", "risk_grade", "risk_level", "risk_o_meter", "risk"]
        if c in df_fund.columns
    ]
    merged = df_perf.merge(
        df_fund[fund_cols],
        on="amfi_code",
        how="inner",
        suffixes=("_perf", ""),
    )
    if "scheme_name" not in merged.columns and "scheme_name_perf" in merged.columns:
        merged["scheme_name"] = merged["scheme_name_perf"]

    risk_col = _resolve_risk_column(merged)
    if risk_col:
        merged["effective_risk_label"] = merged[risk_col].astype(str)
        matched = merged[merged["effective_risk_label"].str.strip().str.lower().isin(valid_tiers[key])]
        if len(matched) < top_n:
            merged["_tier"] = merged["category"].apply(_infer_risk_tier_from_category)
            matched = merged[merged["_tier"].str.lower() == key]
    else:
        merged["_tier"] = merged["category"].apply(_infer_risk_tier_from_category)
        merged["effective_risk_label"] = merged["_tier"]
        matched = merged[merged["_tier"].str.lower() == key]

    if matched.empty:
        matched = merged

    sharpe_col = next((c for c in ["sharpe_ratio", "sharpe_3yr", "sharpe"] if c in matched.columns), None)
    sort_cols = [c for c in [sharpe_col, "return_3yr_pct", "return_1yr_pct"] if c and c in matched.columns]
    recs = (
        matched.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        .head(top_n)
        .reset_index(drop=True)
    )
    recs.insert(0, "rank", recs.index + 1)

    display_cols = [
        c for c in [
            "rank", "amfi_code", "scheme_name", "category", "effective_risk_label",
            sharpe_col, "return_1yr_pct", "return_3yr_pct", "std_dev_ann_pct", "expense_ratio_pct"
        ] if c and c in recs.columns
    ]
    return recs[display_cols].rename(columns={"effective_risk_label": "risk_grade"})


def print_recommendation_table(risk_appetite: str, top_n: int = 3) -> None:
    recs = recommend_funds(risk_appetite, top_n=top_n)
    print(f"\n=== Top {top_n} Recommended Funds for '{risk_appetite.strip().title()}' Risk Appetite ===")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    user_risk = sys.argv[1] if len(sys.argv) > 1 else input("Enter risk appetite (Low / Moderate / High): ")
    print_recommendation_table(user_risk, top_n=3)
