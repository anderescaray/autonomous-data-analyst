"""Run once to generate sample_data.xlsx for testing the app."""
import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)
n = 200

regions    = rng.choice(["North", "South", "East", "West"], n)
categories = rng.choice(["Electronics", "Clothing", "Food", "Sports", "Books"], n)
sales      = (rng.normal(loc=5000, scale=1500, size=n)).clip(500).round(2)
units      = rng.integers(1, 120, size=n)
profit     = (sales * rng.uniform(0.1, 0.4, size=n)).round(2)
rating     = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.20, 0.35, 0.30])
dates      = pd.date_range("2023-01-01", periods=n, freq="2D")

df = pd.DataFrame({
    "date":      dates,
    "region":    regions,
    "category":  categories,
    "sales":     sales,
    "units":     units,
    "profit":    profit,
    "rating":    rating,
})

out = Path("sample_data.xlsx")
df.to_excel(out, index=False)
print(f"Created {out.resolve()}  ({len(df)} rows × {len(df.columns)} columns)")
