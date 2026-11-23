import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

if os.path.exists("data/processed/01_fund_master.csv"):
    DATA_DIR, CHART_DIR = "data/processed", "reports/charts"
elif os.path.exists("mutual-fund-analytics/data/processed/01_fund_master.csv"):
    DATA_DIR, CHART_DIR = "mutual-fund-analytics/data/processed", "mutual-fund-analytics/reports/charts"
else:
    raise FileNotFoundError("Could not find data/processed/01_fund_master.csv")

os.makedirs(CHART_DIR, exist_ok=True)

df_fund  = pd.read_csv(f"{DATA_DIR}/01_fund_master.csv")
df_nav   = pd.read_csv(f"{DATA_DIR}/02_nav_history.csv")
df_aum   = pd.read_csv(f"{DATA_DIR}/03_aum_by_fund_house.csv")
df_sip   = pd.read_csv(f"{DATA_DIR}/04_monthly_sip_inflows.csv")
df_cat   = pd.read_csv(f"{DATA_DIR}/05_category_inflows.csv")
df_folio = pd.read_csv(f"{DATA_DIR}/06_industry_folio_count.csv")
df_perf  = pd.read_csv(f"{DATA_DIR}/07_scheme_performance.csv")
df_txn   = pd.read_csv(f"{DATA_DIR}/08_investor_transactions.csv")
df_hold  = pd.read_csv(f"{DATA_DIR}/09_portfolio_holdings.csv")
df_bench = pd.read_csv(f"{DATA_DIR}/10_benchmark_indices.csv")

nav_m = df_nav.merge(df_fund[["amfi_code", "scheme_name", "category", "sub_category"]], on="amfi_code", how="left")
nav_m["date"] = pd.to_datetime(nav_m["date"])
nav_m = nav_m.sort_values(["amfi_code", "date"])
nav_m["norm_nav"] = nav_m.groupby("amfi_code")["nav"].transform(lambda s: (s / s.iloc[0]) * 100.0)

fig, ax = plt.subplots(figsize=(14, 6))
for _, grp in nav_m.groupby("amfi_code"):
    ax.plot(grp["date"], grp["norm_nav"], alpha=0.35, linewidth=0.9)
ax.plot(nav_m.groupby("date")["norm_nav"].mean(), color="#0B2545", linewidth=2.4, label="40-Scheme Mean (Base=100)")
ax.axvspan(pd.Timestamp("2023-04-01"), pd.Timestamp("2023-12-31"), color="#2A9D8F", alpha=0.16, label="2023 Bull Run")
ax.axvspan(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-11-30"), color="#E76F51", alpha=0.16, label="2024 Corrections")
ax.set_title("Chart 01: Normalized Daily NAV Across All 40 Schemes (2022–2026)", fontweight="bold")
ax.set_ylabel("Normalized NAV (Jan 2022 = 100)")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/01_nav_trend_all_40_schemes.png", dpi=180); plt.close()

a = df_aum.copy(); a["year"] = pd.to_datetime(a["date"]).dt.year
ay = a[a["year"].between(2022, 2025)].sort_values("date").groupby(["fund_house", "year"], as_index=False).last()
ord_fh = ay[ay["year"] == 2025].sort_values("aum_lakh_crore", ascending=False)["fund_house"].tolist()
fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(data=ay, x="fund_house", y="aum_lakh_crore", hue="year", order=ord_fh, palette=["#8ECAE6", "#219EBC", "#126782", "#023047"], ax=ax)
sbi_v = ay[(ay["fund_house"] == "SBI Mutual Fund") & (ay["year"] == 2025)]["aum_lakh_crore"].values[0]
b = ax.containers[3][0]; b.set_facecolor("#FFB703"); b.set_edgecolor("#B23A48"); b.set_linewidth(1.8)
ax.annotate(f"SBI (2025): ₹{sbi_v:.2f}L Cr", xy=(b.get_x() + b.get_width()/2, sbi_v), xytext=(0.7, sbi_v - 0.8),
            arrowprops=dict(arrowstyle="->", color="#B23A48", lw=1.6), fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="#FFB703"))
ax.set_title("Chart 02: Year-End AUM by Fund House (2022–2025) — SBI Dominance", fontweight="bold")
plt.xticks(rotation=25, ha="right"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/02_aum_growth_by_fund_house.png", dpi=180); plt.close()

s = df_sip.sort_values("month").copy(); pk = s.loc[s["sip_inflow_crore"].idxmax()]
fig, ax = plt.subplots(figsize=(14, 5.5)); ax.plot(s["month"], s["sip_inflow_crore"], marker="o", lw=2.2, color="#1D3557")
ax.scatter([pk["month"]], [pk["sip_inflow_crore"]], color="#E63946", s=90, zorder=5)
ax.annotate(f"Peak ({pk['month']}): ₹{int(pk['sip_inflow_crore']):,} Cr", xy=(pk["month"], pk["sip_inflow_crore"]),
            xytext=(34, pk["sip_inflow_crore"] - 2500), arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.6), fontweight="bold")
ax.set_title("Chart 03: Monthly SIP Inflows (Jan 2022 – Dec 2025)", fontweight="bold"); plt.xticks(s["month"][::3], rotation=35, ha="right")
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/03_monthly_sip_inflows.png", dpi=180); plt.close()

cp = df_cat.pivot(index="category", columns="month", values="net_inflow_crore")
cp = cp.loc[cp.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(13, 6.5)); sns.heatmap(cp, cmap="RdYlGn", center=0, annot=True, fmt=".0f", linewidths=0.5, ax=ax)
ax.set_title("Chart 04: Monthly Net Inflows by Category (₹ Cr)", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/04_category_inflow_heatmap.png", dpi=180); plt.close()

ao = ["18-25", "26-35", "36-45", "46-55", "56+"]
fig, ax = plt.subplots(figsize=(7.5, 6)); ax.pie(df_txn["age_group"].value_counts().reindex(ao), labels=ao, autopct="%1.1f%%", startangle=140)
ax.set_title("Chart 05A: Investor Share by Age Group", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/05a_age_group_pie.png", dpi=180); plt.close()

sip_d = df_txn[df_txn["transaction_type"] == "SIP"]
fig, ax = plt.subplots(figsize=(10, 5.5)); sns.boxplot(data=sip_d, x="age_group", y="amount_inr", order=ao, palette="Blues", ax=ax)
ax.set_title("Chart 05B: SIP Amount by Age Group (INR)", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/05b_sip_amount_by_age_boxplot.png", dpi=180); plt.close()

gs = df_txn.groupby("gender", as_index=False).agg(cnt=("investor_id", "count"), cr=("amount_inr", lambda x: x.sum() / 1e7))
fig, axes = plt.subplots(1, 2, figsize=(11, 5)); axes[0].pie(gs["cnt"], labels=gs["gender"], autopct="%1.1f%%", startangle=90); axes[0].set_title("Count by Gender")
sns.barplot(data=gs, x="gender", y="cr", ax=axes[1]); axes[1].set_title("Volume by Gender (₹ Cr)"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/05c_gender_split.png", dpi=180); plt.close()

sts = sip_d.groupby("state", as_index=False)["amount_inr"].sum().assign(cr=lambda d: d["amount_inr"] / 1e7).sort_values("cr", ascending=False)
fig, ax = plt.subplots(figsize=(11, 6)); sns.barplot(data=sts, x="cr", y="state", palette="viridis", ax=ax)
ax.set_title("Chart 06A: SIP Volume by State (₹ Cr)", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/06a_sip_amount_by_state.png", dpi=180); plt.close()

tc = df_txn["city_tier"].value_counts()
fig, ax = plt.subplots(figsize=(7, 6)); ax.pie(tc, labels=[f"{k} Cities" for k in tc.index], autopct="%1.1f%%", startangle=110)
ax.set_title("Chart 06B: T30 vs B30 Share", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/06b_t30_vs_b30_pie.png", dpi=180); plt.close()

fs = df_folio.sort_values("month").copy()
fig, ax = plt.subplots(figsize=(13, 5.5)); ax.plot(fs["month"], fs["total_folios_crore"], marker="o", lw=2.5, label="Total Folios (Cr)")
ax.plot(fs["month"], fs["equity_folios_crore"], linestyle="--", marker="s", label="Equity Folios (Cr)")
for m_s, l in [("2022-01", "13.26 Cr"), ("2023-04", "15.54 Cr"), ("2024-10", "21.62 Cr"), ("2025-12", "26.12 Cr")]:
    r = fs[fs["month"] == m_s].iloc[0]; ax.scatter([r["month"]], [r["total_folios_crore"]], color="#E63946", s=65, zorder=5)
    ax.annotate(f"{m_s}: {l}", (r["month"], r["total_folios_crore"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
ax.set_ylim(8, 29); ax.set_title("Chart 07: Industry Folio Count Growth", fontweight="bold"); plt.xticks(rotation=35, ha="right"); ax.legend()
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/07_folio_count_growth.png", dpi=180); plt.close()

t10 = df_perf.sort_values("aum_crore", ascending=False).head(10)["amfi_code"].tolist()
nm = df_fund[df_fund["amfi_code"].isin(t10)].set_index("amfi_code")["scheme_name"].str.replace(" - Regular Plan - Growth", "", regex=False).str.replace(" - Regular - Growth", "", regex=False).to_dict()
piv = df_nav[df_nav["amfi_code"].isin(t10)].pivot(index="date", columns="amfi_code", values="nav").sort_index()
ret = piv.pct_change().dropna(); ret.columns = [nm[c] for c in ret.columns]
fig, ax = plt.subplots(figsize=(11, 8.5)); sns.heatmap(ret.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.2, vmax=1.0, linewidths=0.5, ax=ax)
ax.set_title("Chart 08: Daily NAV Return Correlation (Top 10)", fontweight="bold"); plt.xticks(rotation=35, ha="right")
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/08_nav_return_correlation.png", dpi=180); plt.close()

eh = df_hold.merge(df_fund[["amfi_code", "category"]], on="amfi_code").query("category == 'Equity'")
sec = eh.groupby("sector", as_index=False)["weight_pct"].sum().sort_values("weight_pct", ascending=False); sec["pct"] = sec["weight_pct"] / sec["weight_pct"].sum() * 100
maj = sec[sec["pct"] >= 3.5].copy(); oth = sec[sec["pct"] < 3.5]["pct"].sum()
if oth > 0: maj = pd.concat([maj, pd.DataFrame([{"sector": "Others", "pct": oth}])], ignore_index=True)
fig, ax = plt.subplots(figsize=(9, 7)); ax.pie(maj["pct"], labels=maj["sector"], autopct="%1.1f%%", startangle=90, pctdistance=0.80, wedgeprops={"width": 0.42, "edgecolor": "white"})
ax.text(0, 0, "Equity\nSectors", ha="center", va="center", fontweight="bold"); ax.set_title("Chart 09: Equity Sector Allocation Donut", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/09_sector_allocation_donut.png", dpi=180); plt.close()

fig, ax = plt.subplots(figsize=(10, 6)); sns.scatterplot(data=df_perf, x="std_dev_ann_pct", y="return_3yr_pct", hue="category", size="aum_crore", ax=ax)
ax.set_title("Chart 10: 3Y Return vs Volatility", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/10_risk_return_scatter.png", dpi=180); plt.close()

fig, ax = plt.subplots(figsize=(9, 5.5)); sns.violinplot(data=df_fund, x="category", y="expense_ratio_pct", hue="plan", split=True, ax=ax)
ax.set_title("Chart 11: Expense Ratio Direct vs Regular", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/11_expense_ratio_direct_vs_regular.png", dpi=180); plt.close()

ta = df_perf.sort_values("alpha", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(11, 5.5)); sns.barplot(data=ta, x="alpha", y="scheme_name", ax=ax)
ax.set_title("Chart 12: Top 10 Schemes by Alpha", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/12_top_schemes_by_alpha.png", dpi=180); plt.close()

b = df_bench.copy(); b["date"] = pd.to_datetime(b["date"]); b = b.sort_values(["index_name", "date"]); b["norm"] = b.groupby("index_name")["close_value"].transform(lambda s: s / s.iloc[0] * 100)
fig, ax = plt.subplots(figsize=(13, 5.5)); sns.lineplot(data=b, x="date", y="norm", hue="index_name", ax=ax)
ax.set_title("Chart 13: Benchmark Indices (Base=100)", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/13_benchmark_indices_trend.png", dpi=180); plt.close()

fig, ax = plt.subplots(figsize=(9, 5.5)); pd.crosstab(df_txn["payment_mode"], df_txn["transaction_type"]).plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Chart 14: Payment Mode Mix", fontweight="bold"); plt.tight_layout(); plt.savefig(f"{CHART_DIR}/14_payment_mode_stacked_bar.png", dpi=180); plt.close()

t2 = df_txn.copy(); t2["ym"] = pd.to_datetime(t2["transaction_date"]).dt.to_period("M").astype(str)
fm = t2.pivot_table(index="ym", columns="transaction_type", values="amount_inr", aggfunc="sum", fill_value=0) / 1e7
fm["Net"] = fm.get("SIP", 0) + fm.get("Lumpsum", 0) - fm.get("Redemption", 0)
fig, ax = plt.subplots(figsize=(13, 5.5)); ax.bar(fm.index, fm["Net"], alpha=0.8, label="Net (₹ Cr)"); ax.plot(fm.index, fm["SIP"], marker="o", color="#1D3557", label="SIP (₹ Cr)")
ax.set_title("Chart 15: Monthly Net vs SIP Flows", fontweight="bold"); plt.xticks(fm.index[::2], rotation=35, ha="right"); ax.legend()
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/15_monthly_net_vs_sip_flows.png", dpi=180); plt.close()

print("DONE:", len([f for f in os.listdir(CHART_DIR) if f.endswith(".png")]), "PNG charts saved in", CHART_DIR)
