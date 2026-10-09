"""
Loads the cleaned customer-month panel and the industry benchmark reference
table into PostgreSQL (database: churn_retention).
"""
import pandas as pd
from pathlib import Path

from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Change YOUR_PASSWORD to the password you created for PostgreSQL
ENGINE_URL = "postgresql+psycopg2://postgres:Space#7789@localhost:5432/churn_retention"

engine = create_engine(ENGINE_URL)

panel = pd.read_csv(
    PROJECT_ROOT / "data" / "processed" / "customer_month_clean.csv",
    parse_dates=["activity_month", "signup_month"]
)

benchmarks = pd.read_csv(
    PROJECT_ROOT / "data" / "reference" / "industry_benchmarks.csv"
)

panel.to_sql("customer_month", engine, if_exists="replace", index=False, chunksize=10000)
benchmarks.to_sql("industry_benchmarks", engine, if_exists="replace", index=False)

with engine.connect() as conn:
    conn.execute(text("CREATE INDEX idx_cm_customer ON customer_month(customer_id);"))
    conn.execute(text("CREATE INDEX idx_cm_month ON customer_month(activity_month);"))
    conn.execute(text("CREATE INDEX idx_cm_customer_month ON customer_month(customer_id, activity_month);"))
    conn.execute(text("CREATE INDEX idx_cm_cohort ON customer_month(cohort_year, cohort_month);"))
    conn.commit()

print(f"Loaded customer_month: {len(panel)} rows, {panel['customer_id'].nunique()} unique customers")
print(f"Loaded industry_benchmarks: {len(benchmarks)} rows")
