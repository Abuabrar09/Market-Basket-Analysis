-- =====================================================================
-- 01_schema.sql  |  Instacart Market Basket schema (SQLite / ANSI SQL)
-- =====================================================================
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS aisles;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS order_products;

CREATE TABLE departments (
    department_id   INTEGER PRIMARY KEY,
    department      TEXT NOT NULL
);

CREATE TABLE aisles (
    aisle_id        INTEGER PRIMARY KEY,
    aisle           TEXT NOT NULL
);

CREATE TABLE products (
    product_id      INTEGER PRIMARY KEY,
    product_name    TEXT NOT NULL,
    aisle_id        INTEGER REFERENCES aisles(aisle_id),
    department_id   INTEGER REFERENCES departments(department_id)
);

CREATE TABLE orders (
    order_id                INTEGER PRIMARY KEY,
    user_id                 INTEGER NOT NULL,
    eval_set                TEXT,
    order_number            INTEGER,
    order_dow               INTEGER,   -- 0..6
    order_hour_of_day       INTEGER,   -- 0..23
    days_since_prior_order  REAL
);

CREATE TABLE order_products (
    order_id            INTEGER REFERENCES orders(order_id),
    product_id          INTEGER REFERENCES products(product_id),
    add_to_cart_order   INTEGER,
    reordered           INTEGER
);
