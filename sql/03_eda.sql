-- =====================================================================
-- 03_eda.sql  |  Exploratory analysis. Each query is named with
-- "-- name: <query_name>" so the Python pipeline can run it and
-- export the result for Tableau.
-- =====================================================================

-- name: kpi_overview
SELECT
    (SELECT COUNT(*)                   FROM orders)          AS total_orders,
    (SELECT COUNT(DISTINCT user_id)    FROM orders)          AS total_customers,
    (SELECT COUNT(*)                   FROM order_products)  AS total_items_sold,
    (SELECT COUNT(DISTINCT product_id) FROM order_products)  AS distinct_products,
    ROUND((SELECT COUNT(*) FROM order_products) * 1.0 /
          (SELECT COUNT(*) FROM orders), 2)                  AS avg_basket_size,
    ROUND((SELECT AVG(reordered) FROM order_products), 4)    AS reorder_rate;

-- name: top_products
SELECT  p.product_id,
        p.product_name,
        d.department,
        a.aisle,
        COUNT(*)                                   AS orders_with_product,
        ROUND(COUNT(*) * 1.0 / (SELECT COUNT(*) FROM orders), 4) AS support,
        ROUND(AVG(op.reordered), 4)                AS reorder_rate,
        ROUND(AVG(op.add_to_cart_order), 2)        AS avg_cart_position
FROM order_products op
JOIN products    p ON p.product_id    = op.product_id
JOIN departments d ON d.department_id = p.department_id
JOIN aisles      a ON a.aisle_id      = p.aisle_id
GROUP BY p.product_id, p.product_name, d.department, a.aisle
ORDER BY orders_with_product DESC;

-- name: department_summary
SELECT  d.department,
        COUNT(*)                                        AS items_sold,
        COUNT(DISTINCT op.order_id)                     AS orders_touching_dept,
        ROUND(COUNT(DISTINCT op.order_id) * 1.0 /
              (SELECT COUNT(*) FROM orders), 4)         AS order_penetration,
        ROUND(AVG(op.reordered), 4)                     AS reorder_rate
FROM order_products op
JOIN products    p ON p.product_id    = op.product_id
JOIN departments d ON d.department_id = p.department_id
GROUP BY d.department
ORDER BY items_sold DESC;

-- name: orders_by_dow_hour
SELECT  order_dow,
        CASE order_dow WHEN 0 THEN 'Sun' WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue'
                       WHEN 3 THEN 'Wed' WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri'
                       ELSE 'Sat' END                AS day_name,
        order_hour_of_day,
        COUNT(*)                                     AS orders
FROM orders
GROUP BY order_dow, order_hour_of_day
ORDER BY order_dow, order_hour_of_day;

-- name: basket_size_distribution
WITH sizes AS (
    SELECT order_id, COUNT(*) AS basket_size
    FROM order_products
    GROUP BY order_id
)
SELECT basket_size, COUNT(*) AS orders
FROM sizes
GROUP BY basket_size
ORDER BY basket_size;

-- name: first_in_cart_products
-- "Anchor" products: what shoppers add FIRST - strong trigger points for cross-sell prompts
SELECT  p.product_name,
        COUNT(*)                                            AS times_added_first,
        ROUND(COUNT(*) * 1.0 / (SELECT COUNT(*) FROM orders), 4) AS share_of_orders
FROM order_products op
JOIN products p ON p.product_id = op.product_id
WHERE op.add_to_cart_order = 1
GROUP BY p.product_name
ORDER BY times_added_first DESC
LIMIT 25;
