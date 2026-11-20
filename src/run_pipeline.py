"""
run_pipeline.py — End-to-end Market Basket & Product Association Analysis.

  1. Load Instacart-schema CSVs into a SQLite database        (SQL)
  2. Run exploratory SQL queries + SQL-only pair rules         (SQL)
  3. Mine frequent itemsets with Apriori + association rules   (Python)
  4. Cross-check Python lift against SQL lift                  (QA)
  5. Score cross-selling opportunities                         (Business)
  6. Export Tableau-ready CSVs, charts and metrics.json        (Tableau)

Usage
  python src/run_pipeline.py                         # sample data (default)
  python src/run_pipeline.py --data-dir data/raw --sample-orders 200000   # real Instacart
"""
import argparse
import json
import os
import re
import sqlite3
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

from apriori import association_rules, frequent_itemsets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_DIR = os.path.join(ROOT, "sql")
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tableau")

# ---- chart style (validated reference palette; one blue for magnitude) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a8", "#e66767", "#b8302f"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.family": "DejaVu Sans",
})


# ----------------------------------------------------------------------------
# 1. SQL helpers
# ----------------------------------------------------------------------------
def read_named_queries(path):
    """Parse a .sql file whose queries are tagged with '-- name: <query>'."""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def load_database(data_dir, db_path, sample_orders=None, seed=7):
    t0 = time.time()
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(open(os.path.join(SQL_DIR, "01_schema.sql")).read())

    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    orders = orders[orders.eval_set == "prior"]          # prior orders hold the baskets
    if sample_orders and len(orders) > sample_orders:
        orders = orders.sample(sample_orders, random_state=seed)
    keep = set(orders.order_id)

    lines = []
    for chunk in pd.read_csv(os.path.join(data_dir, "order_products__prior.csv"), chunksize=2_000_000):
        lines.append(chunk[chunk.order_id.isin(keep)])
    order_products = pd.concat(lines, ignore_index=True)

    for name in ["departments", "aisles", "products"]:
        pd.read_csv(os.path.join(data_dir, f"{name}.csv")).to_sql(name, con, if_exists="append", index=False)
    orders.to_sql("orders", con, if_exists="append", index=False)
    order_products.to_sql("order_products", con, if_exists="append", index=False)
    con.executescript(open(os.path.join(SQL_DIR, "02_indexes.sql")).read())
    con.commit()
    print(f"[load] {len(orders):,} orders | {len(order_products):,} lines -> {db_path} ({time.time()-t0:.1f}s)")
    return con


# ----------------------------------------------------------------------------
# 2. Charts
# ----------------------------------------------------------------------------
def _finish(ax, fig, path, xlabel=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_top_products(top, path):
    d = top.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d.product_name, d.support * 100, color=BLUE, height=0.6)
    for y, v in enumerate(d.support * 100):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", color=INK2, fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Top 15 products by share of orders")
    _finish(ax, fig, path, "% of orders containing product")


def unique_pairs(rules):
    """Lift is symmetric, so keep one direction per pair (the higher-confidence one)."""
    d = rules.copy()
    d["pair_key"] = [tuple(sorted(a + c)) for a, c in zip(d.antecedent_ids, d.consequent_ids)]
    return d.sort_values("confidence", ascending=False).drop_duplicates("pair_key")


def chart_top_rules(rules, path):
    d = unique_pairs(rules).sort_values("lift", ascending=False).head(15).iloc[::-1]
    labels = [f"{a}  →  {c}" for a, c in zip(d.antecedent, d.consequent)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(labels, d.lift, color=BLUE, height=0.6)
    for y, (l, c) in enumerate(zip(d.lift, d.confidence)):
        ax.text(l + 0.1, y, f"lift {l:.1f} · conf {c:.0%}", va="center", color=INK2, fontsize=8.5)
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Strongest product associations (pair rules, ranked by lift)")
    _finish(ax, fig, path, "Lift (1.0 = bought together by chance)")


def chart_rule_scatter(rules, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", SEQ_BLUES[1:])
    sc = ax.scatter(rules.support * 100, rules.confidence * 100, c=rules.lift, cmap=cmap,
                    s=28, edgecolors=SURFACE, linewidths=0.6)
    cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("Lift"); cb.outline.set_visible(False)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_ylabel("Confidence (%)")
    ax.set_title(f"All {len(rules):,} association rules: support vs confidence")
    _finish(ax, fig, path, "Support (% of orders)")


def chart_dept_heatmap(dept, path):
    m = dept.pivot(index="department_a", columns="department_b", values="lift")
    order = m.mean(axis=1).sort_values(ascending=False).index
    m = m.loc[order, order]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    cmap = mcolors.LinearSegmentedColormap.from_list("div", DIVERGING)
    norm = mcolors.TwoSlopeNorm(vmin=min(0.5, np.nanmin(m.values)), vcenter=1.0,
                                vmax=max(1.5, np.nanmax(m.values)))
    im = ax.imshow(m.values, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Lift (blue < 1 < red)"); cb.outline.set_visible(False)
    ax.set_title("Department affinity: which aisles are shopped together")
    _finish(ax, fig, path)


def chart_dow_hour(dh, path):
    m = dh.pivot(index="day_name", columns="order_hour_of_day", values="orders").reindex(
        ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", ["#f0efec"] + SEQ_BLUES)
    im = ax.imshow(m.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)], fontsize=8.5)
    ax.set_yticks(range(7), m.index)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("Orders"); cb.outline.set_visible(False)
    ax.set_title("When customers shop: orders by day and hour")
    _finish(ax, fig, path)


def chart_network(rules, path, n=45):
    """Undirected affinity network; each connected cluster is laid out in its own grid cell."""
    d = unique_pairs(rules).sort_values("lift", ascending=False).head(n)
    g = nx.Graph()
    for r in d.itertuples():
        g.add_edge(r.antecedent, r.consequent, weight=r.lift)
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    cols = 4
    rows = int(np.ceil(len(comps) / cols))
    pos, cy, total_h = {}, 0.0, 0.0
    for r_ in range(rows):
        row = comps[r_ * cols:(r_ + 1) * cols]
        row_h = 2.9 if max(len(c) for c in row) > 2 else 1.1
        cy -= row_h / 2
        for i, comp in enumerate(row):
            sub = g.subgraph(comp)
            p = nx.circular_layout(sub) if len(comp) > 2 else {k: np.array([(j - 0.5) * 1.7, 0]) for j, k in enumerate(comp)}
            cx = i * 3.2
            for k, (x, y) in p.items():
                pos[k] = np.array([cx + x * 0.95, cy + y * 0.95])
        cy -= row_h / 2
        total_h += row_h
    fig, ax = plt.subplots(figsize=(14, 1.15 * total_h + 1.2))
    w = np.array([g[u][v]["weight"] for u, v in g.edges])
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=AXIS, width=0.8 + 3 * (w - w.min()) / (np.ptp(w) + 1e-9))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=BLUE, node_size=160, edgecolors=SURFACE, linewidths=2)
    for node, (x, y) in pos.items():
        label = node if len(node) <= 22 else node[:21] + "…"
        ax.text(x, y - 0.24, label, ha="center", va="top", fontsize=7.8, color=INK,
                bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.9))
    ax.set_axis_off()
    ax.margins(0.06)
    ax.set_title(f"Product affinity clusters (top {n} product pairs by lift; thicker line = higher lift)")
    _finish(ax, fig, path)


# ----------------------------------------------------------------------------
# 3. Pipeline
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "sample"))
    ap.add_argument("--sample-orders", type=int, default=None,
                    help="random subset of prior orders (use ~200000 for the full Instacart data)")
    ap.add_argument("--min-support", type=float, default=0.005)
    ap.add_argument("--min-confidence", type=float, default=0.10)
    ap.add_argument("--min-lift", type=float, default=1.2)
    ap.add_argument("--attach-rate", type=float, default=0.05,
                    help="assumed conversion of a 'frequently bought together' prompt")
    ap.add_argument("--dataset-label", default=None)
    args = ap.parse_args()
    for p in (OUT, FIG, TAB):
        os.makedirs(p, exist_ok=True)

    con = load_database(args.data_dir, os.path.join(OUT, "market_basket.db"), args.sample_orders)

    # ---------- SQL: EDA ----------
    eda = read_named_queries(os.path.join(SQL_DIR, "03_eda.sql"))
    res = {name: pd.read_sql_query(q, con) for name, q in eda.items()}
    kpi = res["kpi_overview"].iloc[0].to_dict()
    print("[sql] KPIs:", kpi)

    # ---------- SQL: pair rules + department affinity ----------
    aff = read_named_queries(os.path.join(SQL_DIR, "04_pair_affinity.sql"))
    n_orders = int(kpi["total_orders"])
    min_pair = max(5, int(args.min_support * n_orders))
    t0 = time.time()
    sql_pairs = pd.read_sql_query(aff["pair_rules_sql"], con, params={"min_pair_orders": min_pair})
    dept = pd.read_sql_query(aff["department_affinity"], con)
    print(f"[sql] pair rules: {len(sql_pairs):,} ({time.time()-t0:.1f}s)")

    # ---------- Python: Apriori ----------
    t0 = time.time()
    op = pd.read_sql_query("SELECT order_id, product_id FROM order_products", con)
    baskets = [frozenset(s) for s in op.groupby("order_id").product_id.agg(set)]
    itemsets, n = frequent_itemsets(baskets, min_support=args.min_support, max_len=3)
    rules = association_rules(itemsets, n, args.min_confidence, args.min_lift)
    names = pd.read_sql_query(
        """SELECT p.product_id, p.product_name, d.department
           FROM products p JOIN departments d USING (department_id)""", con)
    pname = dict(zip(names.product_id, names.product_name))
    pdept = dict(zip(names.product_id, names.department))
    rules["antecedent"] = rules.antecedent_ids.map(lambda t: " + ".join(pname[i] for i in t))
    rules["consequent"] = rules.consequent_ids.map(lambda t: " + ".join(pname[i] for i in t))
    rules["antecedent_department"] = rules.antecedent_ids.map(lambda t: pdept[t[0]])
    rules["consequent_department"] = rules.consequent_ids.map(lambda t: pdept[t[0]])
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    print(f"[apriori] {len(itemsets):,} frequent itemsets | {len(rules):,} rules ({time.time()-t0:.1f}s)")

    # ---------- QA: Python lift == SQL lift for every pair rule ----------
    pair_py = rules[rules.rule_size == 2][["antecedent", "consequent", "lift"]]
    chk = pair_py.merge(sql_pairs, on=["antecedent", "consequent"], suffixes=("_py", "_sql"))
    max_diff = float((chk.lift_py - chk.lift_sql).abs().max()) if len(chk) else float("nan")
    print(f"[qa] {len(chk):,} pair rules matched between SQL and Python | max |Δlift| = {max_diff:.4f}")
    assert len(chk) == len(pair_py) and max_diff < 0.01, "SQL and Python association rules disagree!"

    # ---------- Cross-selling opportunities ----------
    support_cnt = {next(iter(k)): v for k, v in itemsets.items() if len(k) == 1}
    pairs = rules[(rules.rule_size == 2) & (rules.lift >= 1.5) & (rules.confidence >= 0.15)].copy()
    pairs["anchor_orders"] = pairs.antecedent_ids.map(lambda t: support_cnt[t[0]])
    pairs["untapped_orders"] = pairs.anchor_orders - pairs.orders_together  # bought A but not B
    pairs["est_incremental_units"] = (pairs.untapped_orders * args.attach_rate).round().astype(int)
    pairs["opportunity_score"] = pairs.untapped_orders * pairs.confidence * np.log(pairs.lift)
    pairs = pairs.sort_values("opportunity_score", ascending=False)
    pairs["recommendation_rank"] = pairs.groupby("antecedent").cumcount() + 1
    cross_sell = pairs[pairs.recommendation_rank <= 3]
    cross_sell = cross_sell.assign(
        cross_dept=np.where(cross_sell.antecedent_department != cross_sell.consequent_department, "Cross-department", "Same department"),
        play=lambda d: np.where(d.cross_dept == "Cross-department", "Recipe bundle / 'Complete the meal' prompt",
                                "Shelf adjacency / 'Frequently bought together'"))

    # bundle rules: two items in cart -> recommend the third
    bundles = rules[(rules.rule_size == 3) & (rules.consequent_ids.map(len) == 1)]
    bundles = bundles.sort_values(["lift", "confidence"], ascending=False).head(15)

    # ---------- Exports (Tableau-ready CSVs) ----------
    export_cols = ["antecedent", "consequent", "antecedent_department", "consequent_department", "rule_size",
                   "orders_together", "support", "confidence", "lift", "leverage", "conviction"]
    rules[export_cols].replace([np.inf], 999).to_csv(f"{TAB}/association_rules.csv", index=False)
    cross_sell[["antecedent", "consequent", "antecedent_department", "consequent_department", "cross_dept",
                "recommendation_rank", "anchor_orders", "orders_together", "untapped_orders", "confidence",
                "lift", "est_incremental_units", "opportunity_score", "play"]].to_csv(
        f"{TAB}/cross_sell_opportunities.csv", index=False)
    res["top_products"].to_csv(f"{TAB}/top_products.csv", index=False)
    res["department_summary"].to_csv(f"{TAB}/department_summary.csv", index=False)
    res["orders_by_dow_hour"].to_csv(f"{TAB}/orders_by_dow_hour.csv", index=False)
    res["basket_size_distribution"].to_csv(f"{TAB}/basket_size_distribution.csv", index=False)
    res["first_in_cart_products"].to_csv(f"{TAB}/first_in_cart_products.csv", index=False)
    dept.to_csv(f"{TAB}/department_affinity.csv", index=False)
    sql_pairs.to_csv(f"{TAB}/pair_rules_from_sql.csv", index=False)

    # ---------- Charts ----------
    pair_rules = rules[rules.rule_size == 2]
    chart_top_products(res["top_products"], f"{FIG}/01_top_products.png")
    chart_top_rules(pair_rules, f"{FIG}/02_top_rules_by_lift.png")
    chart_rule_scatter(rules, f"{FIG}/03_rules_support_confidence.png")
    chart_dept_heatmap(dept, f"{FIG}/04_department_affinity.png")
    chart_dow_hour(res["orders_by_dow_hour"], f"{FIG}/05_orders_by_day_hour.png")
    chart_network(pair_rules, f"{FIG}/06_association_network.png")

    # ---------- metrics.json (summary numbers) ----------
    label = args.dataset_label or ("Simulated sample in Instacart schema" if "sample" in args.data_dir
                                   else "Instacart Market Basket Analysis (public dataset)")
    top_cs = cross_sell.head(10)
    metrics = {
        "dataset": label,
        "kpis": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in kpi.items()},
        "params": {"min_support": args.min_support, "min_confidence": args.min_confidence,
                   "min_lift": args.min_lift, "attach_rate": args.attach_rate},
        "n_frequent_itemsets": len(itemsets),
        "n_rules": int(len(rules)),
        "n_pair_rules": int((rules.rule_size == 2).sum()),
        "n_triple_rules": int((rules.rule_size == 3).sum()),
        "qa_max_lift_diff_sql_vs_python": max_diff,
        "top_rules": unique_pairs(pair_rules).sort_values("lift", ascending=False).head(10)[["antecedent", "consequent", "support", "confidence", "lift"]].round(4).to_dict("records"),
        "top_bundles": bundles.head(8)[["antecedent", "consequent", "support", "confidence", "lift"]].round(4).to_dict("records"),
        "top_cross_sell": top_cs[["antecedent", "consequent", "confidence", "lift", "untapped_orders",
                                  "est_incremental_units", "play"]].round(4).to_dict("records"),
        "total_est_incremental_units_top10": int(top_cs.est_incremental_units.sum()),
        "top_dept_pairs": dept[dept.department_a < dept.department_b].head(6).to_dict("records"),
        "first_in_cart": res["first_in_cart_products"].head(5).to_dict("records"),
    }
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print("\nTop 5 cross-sell opportunities:")
    print(top_cs.head(5)[["antecedent", "consequent", "confidence", "lift", "untapped_orders"]].to_string(index=False))
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
