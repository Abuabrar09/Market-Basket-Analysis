# Data

## Option A: Sample data (default, no account needed)
```bash
python src/generate_sample_data.py          # writes data/sample/*.csv
```
This creates simulated orders in the **exact Instacart schema** (same files and columns). Products, aisles and departments mirror Instacart's catalog structure, and baskets are generated from realistic shopping "missions" such as taco night, baby-care restocks and baking.

## Option B: Real Instacart dataset (recommended for final results)
1. Create a free Kaggle account → *Settings → API → Create New Token* (downloads `kaggle.json`).
2. In a Codespace or terminal:
```bash
pip install kaggle
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d psparks/instacart-market-basket-analysis -p data/raw --unzip
ls data/raw   # aisles.csv departments.csv orders.csv order_products__prior.csv products.csv ...
```
Alternatively, download the zip from the Kaggle page in your browser and upload the CSVs into `data/raw/`.

3. Run the pipeline on a random sample of orders (the full prior table has about 32M rows):
```bash
python src/run_pipeline.py --data-dir data/raw --sample-orders 200000 --min-support 0.002
```
Tips:
- `--min-support 0.002` is a good starting point for the real catalog of about 50K products. Raise it if runs are slow, lower it to find more rules.
- `--sample-orders 200000` fits comfortably in a free 2-core Codespace. Increase it if you have more memory.
- `data/raw/` is git-ignored. **Do not commit the raw dataset**, and check the dataset's terms on Kaggle before redistributing it.
