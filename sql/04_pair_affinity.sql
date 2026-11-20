-- =====================================================================
-- 04_pair_affinity.sql | Association rules for product PAIRS in pure SQL
--
--   support(A,B)      = orders with A and B / total orders
--   confidence(A->B)  = support(A,B) / support(A)
--   lift(A->B)        = confidence(A->B) / support(B)
--
--   lift > 1  : bought together more often than chance  -> cross-sell signal
--   lift = 1  : independent
--   lift < 1  : substitutes / rarely together
--
-- :min_pair_orders is a bind parameter (minimum co-occurrence count).
-- =====================================================================

-- name: pair_rules_sql
WITH total AS (
    SELECT COUNT(*) * 1.0 AS n FROM orders
),
item_counts AS (
    SELECT product_id, COUNT(*) AS cnt
    FROM order_products
    GROUP BY product_id
),
pairs AS (                                -- every unordered pair in each basket
    SELECT  a.product_id AS item_a,
            b.product_id AS item_b,
            COUNT(*)     AS pair_cnt
    FROM order_products a
    JOIN order_products b
      ON a.order_id   = b.order_id
     AND a.product_id < b.product_id
    GROUP BY a.product_id, b.product_id
    HAVING COUNT(*) >= :min_pair_orders
),
rules AS (                                -- both directions: A->B and B->A
    SELECT item_a AS antecedent_id, item_b AS consequent_id, pair_cnt FROM pairs
    UNION ALL
    SELECT item_b, item_a, pair_cnt FROM pairs
)
SELECT  pa.product_name                                         AS antecedent,
        pc.product_name                                         AS consequent,
        r.pair_cnt                                              AS orders_together,
        ROUND(r.pair_cnt / t.n, 5)                              AS support,
        ROUND(r.pair_cnt * 1.0 / ia.cnt, 4)                     AS confidence,
        ROUND((r.pair_cnt * 1.0 / ia.cnt) / (ic.cnt / t.n), 3)  AS lift
FROM rules r
CROSS JOIN total t
JOIN item_counts ia ON ia.product_id = r.antecedent_id
JOIN item_counts ic ON ic.product_id = r.consequent_id
JOIN products    pa ON pa.product_id = r.antecedent_id
JOIN products    pc ON pc.product_id = r.consequent_id
ORDER BY lift DESC;

-- name: department_affinity
-- Department-level lift: which departments are shopped together?
WITH total AS (SELECT COUNT(*) * 1.0 AS n FROM orders),
order_dept AS (
    SELECT DISTINCT op.order_id, p.department_id
    FROM order_products op
    JOIN products p ON p.product_id = op.product_id
),
dept_cnt AS (
    SELECT department_id, COUNT(*) AS cnt FROM order_dept GROUP BY department_id
),
dept_pairs AS (
    SELECT a.department_id AS dept_a, b.department_id AS dept_b, COUNT(*) AS pair_cnt
    FROM order_dept a
    JOIN order_dept b ON a.order_id = b.order_id AND a.department_id <> b.department_id
    GROUP BY a.department_id, b.department_id
)
SELECT  da.department                                            AS department_a,
        db.department                                            AS department_b,
        dp.pair_cnt                                              AS orders_together,
        ROUND(dp.pair_cnt * 1.0 / ca.cnt, 4)                     AS confidence,
        ROUND((dp.pair_cnt / t.n) / ((ca.cnt / t.n) * (cb.cnt / t.n)), 3) AS lift
FROM dept_pairs dp
CROSS JOIN total t
JOIN dept_cnt ca    ON ca.department_id = dp.dept_a
JOIN dept_cnt cb    ON cb.department_id = dp.dept_b
JOIN departments da ON da.department_id = dp.dept_a
JOIN departments db ON db.department_id = dp.dept_b
ORDER BY lift DESC;
