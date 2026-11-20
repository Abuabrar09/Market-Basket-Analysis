"""
apriori.py — A from-scratch implementation of the Apriori algorithm and
association-rule mining (support, confidence, lift, leverage, conviction).

Written without external ML libraries so every step is transparent.
Results are cross-checked against the pure-SQL pair rules in
sql/04_pair_affinity.sql (see run_pipeline.py).
"""
from collections import Counter
from itertools import combinations

import pandas as pd


def frequent_itemsets(baskets, min_support=0.01, max_len=3):
    """
    baskets     : list of sets/frozensets of item ids (one per order)
    min_support : minimum fraction of orders that must contain the itemset
    max_len     : largest itemset size to search
    returns     : dict {frozenset(itemset): count}
    """
    n = len(baskets)
    min_count = min_support * n

    # ---- L1: frequent single items
    c1 = Counter(item for b in baskets for item in b)
    current = {frozenset([i]): c for i, c in c1.items() if c >= min_count}
    result = dict(current)

    k = 2
    while current and k <= max_len:
        # items that appear in any frequent (k-1)-itemset; prune baskets to them
        keep = set().union(*current.keys())
        prev = set(current.keys())

        # ---- candidate generation (join step) + Apriori prune step
        prev_sorted = sorted(tuple(sorted(s)) for s in prev)
        candidates = set()
        for i in range(len(prev_sorted)):
            for j in range(i + 1, len(prev_sorted)):
                a, b = prev_sorted[i], prev_sorted[j]
                if a[:-1] != b[:-1]:
                    break  # sorted, so no more joins with this prefix
                cand = frozenset(a) | frozenset(b)
                if all(frozenset(sub) in prev for sub in combinations(cand, k - 1)):
                    candidates.add(cand)
        if not candidates:
            break

        # ---- count candidates
        counts = Counter()
        for b in baskets:
            items = sorted(b & keep)
            if len(items) < k:
                continue
            for combo in combinations(items, k):
                fs = frozenset(combo)
                if fs in candidates:
                    counts[fs] += 1

        current = {s: c for s, c in counts.items() if c >= min_count}
        result.update(current)
        k += 1

    return result, n


def association_rules(itemsets, n_orders, min_confidence=0.1, min_lift=1.0):
    """Generate rules A -> B from frequent itemsets with standard metrics."""
    rows = []
    for itemset, cnt in itemsets.items():
        if len(itemset) < 2:
            continue
        support_ab = cnt / n_orders
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if antecedent not in itemsets or consequent not in itemsets:
                    continue
                support_a = itemsets[antecedent] / n_orders
                support_b = itemsets[consequent] / n_orders
                confidence = support_ab / support_a
                lift = confidence / support_b
                if confidence < min_confidence or lift < min_lift:
                    continue
                leverage = support_ab - support_a * support_b
                conviction = (1 - support_b) / (1 - confidence) if confidence < 1 else float("inf")
                rows.append({
                    "antecedent_ids": tuple(sorted(antecedent)),
                    "consequent_ids": tuple(sorted(consequent)),
                    "rule_size": len(itemset),
                    "orders_together": cnt,
                    "antecedent_support": support_a,
                    "consequent_support": support_b,
                    "support": support_ab,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                })
    return pd.DataFrame(rows)
