"""
Generate a realistic SAMPLE dataset in the exact schema of the public
Instacart Market Basket Analysis dataset (Kaggle), so the full pipeline
runs out-of-the-box without a Kaggle account.

For the real analysis, download the real data (see data/README.md) and run:
    python src/run_pipeline.py --data-dir data/raw

Files produced (same columns as Instacart):
    departments.csv, aisles.csv, products.csv, orders.csv, order_products__prior.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

DEPARTMENTS = {
    1: "frozen", 2: "other", 3: "bakery", 4: "produce", 5: "alcohol",
    6: "international", 7: "beverages", 8: "pets", 9: "dry goods pasta",
    10: "bulk", 11: "personal care", 12: "meat seafood", 13: "pantry",
    14: "breakfast", 15: "canned goods", 16: "dairy eggs", 17: "household",
    18: "babies", 19: "snacks", 20: "deli", 21: "missing",
}

# (product_name, aisle_name, department_id, base_popularity)
PRODUCTS = [
    # produce
    ("Banana", "fresh fruits", 4, 100), ("Bag of Organic Bananas", "fresh fruits", 4, 80),
    ("Organic Strawberries", "fresh fruits", 4, 55), ("Organic Baby Spinach", "packaged vegetables fruits", 4, 50),
    ("Organic Hass Avocado", "fresh fruits", 4, 45), ("Organic Avocado", "fresh fruits", 4, 38),
    ("Large Lemon", "fresh fruits", 4, 35), ("Limes", "fresh fruits", 4, 33),
    ("Strawberries", "fresh fruits", 4, 30), ("Organic Raspberries", "fresh fruits", 4, 28),
    ("Organic Blueberries", "fresh fruits", 4, 26), ("Organic Yellow Onion", "fresh vegetables", 4, 26),
    ("Organic Garlic", "fresh vegetables", 4, 25), ("Organic Zucchini", "fresh vegetables", 4, 20),
    ("Cucumber Kirby", "fresh vegetables", 4, 22), ("Organic Fuji Apple", "fresh fruits", 4, 22),
    ("Apple Honeycrisp Organic", "fresh fruits", 4, 20), ("Organic Grape Tomatoes", "fresh vegetables", 4, 20),
    ("Organic Cilantro", "fresh herbs", 4, 18), ("Jalapeno Peppers", "fresh vegetables", 4, 15),
    ("Organic Lemon", "fresh fruits", 4, 18), ("Yellow Onions", "fresh vegetables", 4, 18),
    ("Organic Red Bell Pepper", "fresh vegetables", 4, 15), ("Carrots", "fresh vegetables", 4, 15),
    ("Organic Romaine Lettuce", "packaged vegetables fruits", 4, 14), ("Seedless Red Grapes", "fresh fruits", 4, 14),
    ("Russet Potatoes", "fresh vegetables", 4, 12), ("Organic Kiwi", "fresh fruits", 4, 10),
    # dairy eggs
    ("Organic Whole Milk", "milk", 16, 40), ("Organic Half & Half", "cream", 16, 22),
    ("Organic Large Extra Fancy Grade AA Brown Eggs", "eggs", 16, 25), ("Large Grade A Eggs", "eggs", 16, 22),
    ("Unsalted Butter", "butter", 16, 18), ("Shredded Mexican Cheese Blend", "packaged cheese", 16, 16),
    ("Sour Cream", "other creams cheeses", 16, 14), ("Sharp Cheddar Cheese Slices", "packaged cheese", 16, 13),
    ("Grated Parmesan", "packaged cheese", 16, 12), ("Shredded Mozzarella", "packaged cheese", 16, 12),
    ("Total 0% Nonfat Plain Greek Yogurt", "yogurt", 16, 18), ("Organic Strawberry Yogurt", "yogurt", 16, 12),
    ("Original Hazelnut Coffee Creamer", "cream", 16, 12), ("Cream Cheese", "other creams cheeses", 16, 12),
    ("Unsweetened Almondmilk", "soy lactosefree", 16, 20),
    # bakery
    ("Organic Whole Wheat Bread", "bread", 3, 18), ("Hamburger Buns", "buns rolls", 3, 10),
    ("Plain Bagels", "breakfast bakery", 3, 10), ("Flour Tortillas", "tortillas flat bread", 3, 12),
    ("Sourdough Bread", "bread", 3, 10), ("Hot Dog Buns", "buns rolls", 3, 7),
    # meat seafood
    ("Boneless Skinless Chicken Breasts", "poultry counter", 12, 18), ("Ground Beef 85% Lean", "meat counter", 12, 14),
    ("Thick Cut Bacon", "hot dogs bacon sausage", 12, 12), ("Beef Hot Dogs", "hot dogs bacon sausage", 12, 7),
    ("Atlantic Salmon Fillet", "seafood counter", 12, 8),
    # pantry / dry goods
    ("Marinara Sauce", "pasta sauce", 9, 12), ("Spaghetti", "dry pasta", 9, 12),
    ("Penne Rigate", "dry pasta", 9, 8), ("Extra Virgin Olive Oil", "oils vinegars", 13, 12),
    ("Creamy Peanut Butter", "spreads", 13, 12), ("Strawberry Jam", "spreads", 13, 8),
    ("Tomato Ketchup", "condiments", 13, 9), ("Yellow Mustard", "condiments", 13, 6),
    ("Medium Salsa", "preserved dips spreads", 13, 12), ("Taco Seasoning", "spices seasonings", 13, 8),
    ("Honey", "honeys syrups nectars", 13, 8), ("Pure Maple Syrup", "honeys syrups nectars", 13, 6),
    ("All-Purpose Flour", "baking ingredients", 13, 6), ("Granulated Sugar", "baking ingredients", 13, 6),
    ("Semi-Sweet Chocolate Chips", "baking ingredients", 13, 5),
    # canned
    ("Organic Black Beans", "canned meals beans", 15, 10), ("Diced Tomatoes", "canned jarred vegetables", 15, 9),
    ("Chicken Noodle Soup", "soup broth bouillon", 15, 8), ("Organic Chicken Broth", "soup broth bouillon", 15, 8),
    # breakfast
    ("Honey Nut Cheerios", "cereal", 14, 12), ("Old Fashioned Oats", "hot cereal pancake mixes", 14, 9),
    ("Buttermilk Pancake Mix", "hot cereal pancake mixes", 14, 6), ("Granola", "granola", 14, 7),
    # beverages
    ("Sparkling Water Grapefruit", "water seltzer sparkling water", 7, 22), ("Spring Water", "water seltzer sparkling water", 7, 18),
    ("Lime Sparkling Water", "water seltzer sparkling water", 7, 15), ("Medium Roast Ground Coffee", "coffee", 7, 12),
    ("100% Orange Juice", "refrigerated", 7, 12), ("Cola Soda 12 Pack", "soft drinks", 7, 10),
    ("Green Tea", "tea", 7, 6),
    # snacks
    ("Tortilla Chips", "chips pretzels", 19, 14), ("Original Potato Chips", "chips pretzels", 19, 11),
    ("Classic Hummus", "fresh dips tapenades", 20, 14), ("Pita Chips", "crackers", 19, 7),
    ("Cheddar Bunnies", "crackers", 19, 9), ("Dark Chocolate Bar", "candy chocolate", 19, 8),
    ("Mixed Nuts", "nuts seeds dried fruit", 19, 7), ("Microwave Popcorn", "popcorn jerky", 19, 7),
    ("Fruit Snacks", "fruit vegetable snacks", 19, 8),
    # deli
    ("Oven Roasted Turkey Breast", "lunch meat", 20, 10), ("Guacamole", "fresh dips tapenades", 20, 8),
    # frozen
    ("Frozen Organic Blueberries", "frozen produce", 1, 9), ("Vanilla Ice Cream", "ice cream ice", 1, 10),
    ("Frozen Pepperoni Pizza", "frozen pizza", 1, 9), ("Frozen Broccoli Florets", "frozen produce", 1, 8),
    # babies
    ("Baby Wipes Sensitive", "diapers wipes", 18, 6), ("Size 4 Diapers", "diapers wipes", 18, 5),
    ("Organic Stage 2 Pear Baby Food", "baby food formula", 18, 5), ("Organic Baby Puffs", "baby food formula", 18, 4),
    # household / personal care
    ("Paper Towels", "paper goods", 17, 9), ("Toilet Paper 12 Rolls", "paper goods", 17, 8),
    ("Dish Soap", "dish detergents", 17, 6), ("Laundry Detergent", "laundry", 17, 5),
    ("Trash Bags", "trash bags liners", 17, 5), ("Toothpaste", "oral hygiene", 11, 5),
    ("Shampoo", "hair care", 11, 4),
    # pets
    ("Dry Dog Food", "dog food care", 8, 5), ("Dog Treats", "dog food care", 8, 4),
    ("Wet Cat Food", "cat food care", 8, 5), ("Cat Litter", "cat food care", 8, 3),
    # alcohol
    ("IPA Beer 6 Pack", "beers coolers", 5, 6), ("Cabernet Sauvignon", "red wines", 5, 5),
    ("Lager Beer 12 Pack", "beers coolers", 5, 5),
    # international
    ("Soy Sauce", "asian foods", 6, 6), ("Jasmine Rice", "grains rice dried goods", 9, 8),
    ("Coconut Milk", "asian foods", 6, 5),
]

# Shopping "missions" create real co-purchase structure: (items, probability per order)
MISSIONS = [
    (["Organic Strawberries", "Organic Raspberries", "Organic Blueberries", "Bag of Organic Bananas"], 0.10),
    (["Organic Hass Avocado", "Organic Baby Spinach", "Organic Lemon", "Organic Garlic"], 0.08),
    (["Limes", "Organic Cilantro", "Jalapeno Peppers", "Organic Avocado", "Yellow Onions"], 0.06),
    (["Flour Tortillas", "Ground Beef 85% Lean", "Shredded Mexican Cheese Blend", "Sour Cream", "Medium Salsa", "Taco Seasoning"], 0.045),
    (["Tortilla Chips", "Medium Salsa", "Guacamole"], 0.05),
    (["Spaghetti", "Marinara Sauce", "Grated Parmesan", "Organic Garlic"], 0.05),
    (["Penne Rigate", "Marinara Sauce", "Shredded Mozzarella"], 0.025),
    (["Hamburger Buns", "Ground Beef 85% Lean", "Sharp Cheddar Cheese Slices", "Tomato Ketchup", "Yellow Mustard"], 0.035),
    (["Hot Dog Buns", "Beef Hot Dogs", "Tomato Ketchup", "Yellow Mustard"], 0.02),
    (["Large Grade A Eggs", "Thick Cut Bacon", "Sourdough Bread", "Unsalted Butter"], 0.04),
    (["Buttermilk Pancake Mix", "Pure Maple Syrup", "Large Grade A Eggs", "Organic Whole Milk"], 0.02),
    (["Organic Whole Wheat Bread", "Creamy Peanut Butter", "Strawberry Jam"], 0.035),
    (["Plain Bagels", "Cream Cheese"], 0.03),
    (["Medium Roast Ground Coffee", "Organic Half & Half"], 0.035),
    (["Medium Roast Ground Coffee", "Original Hazelnut Coffee Creamer"], 0.025),
    (["Sparkling Water Grapefruit", "Lime Sparkling Water"], 0.07),
    (["Classic Hummus", "Pita Chips", "Cucumber Kirby", "Organic Grape Tomatoes"], 0.035),
    (["Baby Wipes Sensitive", "Size 4 Diapers", "Organic Stage 2 Pear Baby Food", "Organic Baby Puffs"], 0.025),
    (["Paper Towels", "Toilet Paper 12 Rolls", "Trash Bags", "Dish Soap"], 0.03),
    (["Dry Dog Food", "Dog Treats"], 0.02),
    (["Wet Cat Food", "Cat Litter"], 0.015),
    (["Frozen Pepperoni Pizza", "Cola Soda 12 Pack", "Original Potato Chips"], 0.025),
    (["IPA Beer 6 Pack", "Tortilla Chips", "Guacamole"], 0.015),
    (["Boneless Skinless Chicken Breasts", "Jasmine Rice", "Soy Sauce", "Frozen Broccoli Florets"], 0.03),
    (["Atlantic Salmon Fillet", "Organic Lemon", "Organic Zucchini", "Extra Virgin Olive Oil"], 0.015),
    (["All-Purpose Flour", "Granulated Sugar", "Semi-Sweet Chocolate Chips", "Unsalted Butter", "Large Grade A Eggs"], 0.015),
    (["Total 0% Nonfat Plain Greek Yogurt", "Granola", "Organic Blueberries", "Honey"], 0.03),
    (["Chicken Noodle Soup", "Organic Chicken Broth", "Carrots"], 0.015),
    (["Oven Roasted Turkey Breast", "Sharp Cheddar Cheese Slices", "Organic Whole Wheat Bread", "Organic Romaine Lettuce"], 0.025),
    (["Vanilla Ice Cream", "Dark Chocolate Bar"], 0.015),
    (["Cheddar Bunnies", "Fruit Snacks", "Organic Strawberry Yogurt"], 0.03),
]


def build_catalog():
    departments = pd.DataFrame(
        {"department_id": list(DEPARTMENTS), "department": list(DEPARTMENTS.values())}
    )
    aisle_names = sorted({p[1] for p in PRODUCTS})
    aisles = pd.DataFrame({"aisle_id": range(1, len(aisle_names) + 1), "aisle": aisle_names})
    aisle_map = dict(zip(aisles.aisle, aisles.aisle_id))
    products = pd.DataFrame(
        {
            "product_id": range(1, len(PRODUCTS) + 1),
            "product_name": [p[0] for p in PRODUCTS],
            "aisle_id": [aisle_map[p[1]] for p in PRODUCTS],
            "department_id": [p[2] for p in PRODUCTS],
        }
    )
    return departments, aisles, products


def generate(n_users: int, seed: int):
    rng = np.random.default_rng(seed)
    departments, aisles, products = build_catalog()
    name_to_id = dict(zip(products.product_name, products.product_id))
    pop = np.array([p[3] for p in PRODUCTS], dtype=float)
    pop = pop / pop.sum()
    missions = [([name_to_id[n] for n in items], p) for items, p in MISSIONS]

    order_rows, op_rows = [], []
    order_id = 1
    for user_id in range(1, n_users + 1):
        n_orders = int(np.clip(rng.geometric(1 / 12), 4, 60))
        # each user has personal staples they reorder
        staples = rng.choice(len(PRODUCTS), size=rng.integers(3, 9), replace=False, p=pop)
        # users lean toward some missions
        mission_bias = rng.gamma(1.0, 1.0, size=len(missions))
        dow_pref = rng.integers(0, 7)
        hour_pref = int(np.clip(rng.normal(13, 3.5), 6, 22))
        seen = set()
        for k in range(1, n_orders + 1):
            dow = dow_pref if rng.random() < 0.45 else int(rng.choice(7, p=[.19, .18, .13, .12, .12, .13, .13]))
            hour = int(np.clip(round(rng.normal(hour_pref, 2.5)), 0, 23))
            dsp = "" if k == 1 else float(min(30, max(0, round(rng.gamma(2.2, 4.5)))))
            basket = []
            for i, (items, p) in enumerate(missions):
                if rng.random() < min(0.9, p * mission_bias[i]):
                    # shoppers usually buy most (not all) items of a mission
                    basket += [it for it in items if rng.random() < 0.72]
            basket += [int(s) + 1 for s in staples if rng.random() < 0.45]
            n_fill = max(1, int(rng.poisson(5)))
            basket += list(rng.choice(len(PRODUCTS), size=n_fill, p=pop) + 1)
            basket = list(dict.fromkeys(int(b) for b in basket))  # dedupe, keep order
            rng.shuffle(basket)
            order_rows.append((order_id, user_id, "prior", k, dow, hour, dsp))
            for pos, pid in enumerate(basket, start=1):
                op_rows.append((order_id, pid, pos, int(pid in seen)))
                seen.add(pid)
            order_id += 1

    orders = pd.DataFrame(order_rows, columns=[
        "order_id", "user_id", "eval_set", "order_number", "order_dow",
        "order_hour_of_day", "days_since_prior_order"])
    order_products = pd.DataFrame(op_rows, columns=[
        "order_id", "product_id", "add_to_cart_order", "reordered"])
    return departments, aisles, products, orders, order_products


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--users", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d, a, p, o, op = generate(args.users, args.seed)
    d.to_csv(f"{args.out}/departments.csv", index=False)
    a.to_csv(f"{args.out}/aisles.csv", index=False)
    p.to_csv(f"{args.out}/products.csv", index=False)
    o.to_csv(f"{args.out}/orders.csv", index=False)
    op.to_csv(f"{args.out}/order_products__prior.csv", index=False)
    print(f"Sample data written to {args.out}: {len(o):,} orders, "
          f"{len(op):,} order lines, {len(p)} products, {o.user_id.nunique():,} users")


if __name__ == "__main__":
    main()
