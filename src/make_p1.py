import json, os

os.makedirs("notebooks", exist_ok=True)
os.makedirs("reports/charts", exist_ok=True)

def md(text):
    lines = [l + "\n" for l in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(src):
    lines = [l + "\n" for l in src.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = [
    md("# Bluestock Mutual Fund Analytics — Day 3 EDA\nIncludes NAV trends, AUM growth, SIP inflows, category inflows, investor demographics, geographic distribution, industry folios, NAV correlations, equity sector allocation, supplemental charts, and **10 key EDA findings**."),
    code("import os, warnings\nwarnings.filterwarnings('ignore')\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px\nsns.set_theme(style='whitegrid', context='notebook')\nDATA_DIR, CHART_DIR = 'data/processed', 'reports/charts'\nos.makedirs(CHART_DIR, exist_ok=True)\ndf_fund = pd.read_csv(f'{DATA_DIR}/01_fund_master.csv')\ndf_nav = pd.read_csv(f'{DATA_DIR}/02_nav_history.csv')\ndf_aum = pd.read_csv(f'{DATA_DIR}/03_aum_by_fund_house.csv')\ndf_sip = pd.read_csv(f'{DATA_DIR}/04_monthly_sip_inflows.csv')\ndf_cat = pd.read_csv(f'{DATA_DIR}/05_category_inflows.csv')\ndf_folio = pd.read_csv(f'{DATA_DIR}/06_industry_folio_count.csv')\ndf_perf = pd.read_csv(f'{DATA_DIR}/07_scheme_performance.csv')\ndf_txn = pd.read_csv(f'{DATA_DIR}/08_investor_transactions.csv')\ndf_hold = pd.read_csv(f'{DATA_DIR}/09_portfolio_holdings.csv')\ndf_bench = pd.read_csv(f'{DATA_DIR}/10_benchmark_indices.csv')\nprint('Loaded all 10 datasets.')"),
    md("## 1. NAV Trend Analysis (All 40 Schemes, 2022–2026)"),
    code("nav_m = df_nav.merge(df_fund[['amfi_code', 'scheme_name', 'category', 'sub_category']], on='amfi_code', how='left')\nnav_m['date'] = pd.to_datetime(nav_m['date'])\nnav_m = nav_m.sort_values(['amfi_code', 'date'])\nnav_m['norm_nav'] = nav_m.groupby('amfi_code')['nav'].transform(lambda s: (s / s.iloc[0]) * 100.0)\nfig_nav = px.line(nav_m, x='date', y='norm_nav', color='scheme_name', hover_data={'nav': ':.4f', 'category': True}, title='Chart 01: Normalized NAV Across All 40 Schemes (Base=100)')\nfig_nav.add_vrect(x0='2023-04-01', x1='2023-12-31', fillcolor='rgba(42,157,143,0.16)', line_width=0, annotation_text='2023 Bull Run')\nfig_nav.add_vrect(x0='2024-06-01', x1='2024-11-30', fillcolor='rgba(231,111,81,0.16)', line_width=0, annotation_text='2024 Corrections')\nfig_nav.show()\nfig, ax = plt.subplots(figsize=(14, 6))\nfor _, g in nav_m.groupby('amfi_code'): ax.plot(g['date'], g['norm_nav'], alpha=0.35, lw=0.9)\nax.plot(nav_m.groupby('date')['norm_nav'].mean(), color='#0B2545', lw=2.4, label='40-Scheme Mean')\nax.axvspan(pd.Timestamp('2023-04-01'), pd.Timestamp('2023-12-31'), color='#2A9D8F', alpha=0.16, label='2023 Bull Run')\nax.axvspan(pd.Timestamp('2024-06-01'), pd.Timestamp('2024-11-30'), color='#E76F51', alpha=0.16, label='2024 Corrections')\nax.set_title('Chart 01: Normalized Daily NAV (2022–2026)', fontweight='bold'); ax.legend(loc='upper left'); plt.tight_layout(); plt.savefig(f'{CHART_DIR}/01_nav_trend_all_40_schemes.png', dpi=180); plt.close()"),
    md("## 2. AUM Growth by Fund House (2022–2025) — SBI Highlighted"),
    code("a = df_aum.copy(); a['year'] = pd.to_datetime(a['date']).dt.year\nay = a[a['year'].between(2022, 2025)].sort_values('date').groupby(['fund_house', 'year'], as_index=False).last()\nord = ay[ay['year']==2025].sort_values('aum_lakh_crore', ascending=False)['fund_house'].tolist()\nfig, ax = plt.subplots(figsize=(14, 6))\nsns.barplot(data=ay, x='fund_house', y='aum_lakh_crore', hue='year', order=ord, palette=['#8ECAE6','#219EBC','#126782','#023047'], ax=ax)\nsbi_v = ay[(ay['fund_house']=='SBI Mutual Fund') & (ay['year']==2025)]['aum_lakh_crore'].values[0]\nb = ax.containers[3][0]; b.set_facecolor('#FFB703'); b.set_edgecolor('#B23A48'); b.set_linewidth(1.8)\nax.annotate(f'SBI (2025): ₹{sbi_v:.2f}L Cr', xy=(b.get_x()+b.get_width()/2, sbi_v), xytext=(0.7, sbi_v-0.8), arrowprops=dict(arrowstyle='->', color='#B23A48', lw=1.6), fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3CD', ec='#FFB703'))\nax.set_title('Chart 02: Year-End AUM by Fund House (2022–2025)', fontweight='bold'); plt.xticks(rotation=25, ha='right'); plt.tight_layout(); plt.savefig(f'{CHART_DIR}/02_aum_growth_by_fund_house.png', dpi=180); plt.close()"),
    md("## 3. Monthly SIP Inflows (Jan 2022 – Dec 2025)"),
    code("s = df_sip.sort_values('month').copy(); pk = s.loc[s['sip_inflow_crore'].idxmax()]\nfig_s = px.line(s, x='month', y='sip_inflow_crore', markers=True, title='Chart 03: Monthly SIP Inflows')\nfig_s.add_annotation(x=pk['month'], y=pk['sip_inflow_crore'], text=f\"Peak ({pk['month']}): ₹{int(pk['sip_inflow_crore']):,} Cr\", showarrow=True, arrowhead=2)\nfig_s.show()\nfig, ax = plt.subplots(figsize=(14, 5.5)); ax.plot(s['month'], s['sip_inflow_crore'], marker='o', lw=2.2, color='#1D3557')\nax.scatter([pk['month']], [pk['sip_inflow_crore']], color='#E63946', s=90, zorder=5)\nax.annotate(f\"Peak ({pk['month']}): ₹{int(pk['sip_inflow_crore']):,} Cr\", xy=(pk['month'], pk['sip_inflow_crore']), xytext=(34, pk['sip_inflow_crore']-2500), arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.6), fontweight='bold')\nax.set_title('Chart 03: Monthly SIP Inflows (Jan 2022 – Dec 2025)', fontweight='bold'); plt.xticks(s['month'][::3], rotation=35, ha='right'); plt.tight_layout(); plt.savefig(f'{CHART_DIR}/03_monthly_sip_inflows.png', dpi=180); plt.close()")
]
with open("src/build_eda_part1.py", "w", encoding="utf-8") as f:
    json.dump(cells, f)
print("Saved part 1")
