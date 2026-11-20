-- Indexes created AFTER bulk load (much faster than indexing during insert)
CREATE INDEX IF NOT EXISTS ix_op_order   ON order_products(order_id);
CREATE INDEX IF NOT EXISTS ix_op_product ON order_products(product_id);
CREATE INDEX IF NOT EXISTS ix_orders_user ON orders(user_id);
