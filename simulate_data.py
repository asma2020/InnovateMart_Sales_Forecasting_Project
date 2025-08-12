# simulate_data.py
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)

def simulate_innovatemart(start_date="2022-01-01", periods=730, n_stores=4):
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start, periods=periods, freq="D")

    store_ids = [f"store_{i+1}" for i in range(n_stores)]
    store_sizes = ["small", "medium", "large", "medium"][:n_stores]
    city_populations = [50000, 200000, 800000, 150000][:n_stores]

    rows = []
    # create promo calendar (known ahead)
    promo_calendar = {}
    for s in store_ids:
        promo_days = set()
        # periodic: 20% of weekends have promo
        for d in dates:
            if d.weekday() >= 5 and np.random.rand() < 0.2:
                promo_days.add(d)
        # 3 large campaigns per year approx
        campaign_days = [
            start + pd.Timedelta(days=int(periods*0.2)),
            start + pd.Timedelta(days=int(periods*0.5)),
            start + pd.Timedelta(days=int(periods*0.8))
        ]
        for cd in campaign_days:
            for delta in range(-1, 3):
                promo_days.add(pd.to_datetime(cd + pd.Timedelta(days=delta)))
        promo_calendar[s] = promo_days

    # competitor shock near store_2
    competitor_open_date = pd.to_datetime(start + pd.Timedelta(days=int(periods*0.45)))

    for s, size, pop in zip(store_ids, store_sizes, city_populations):
        if size == "small":
            base = 120 + pop * 0.0004
        elif size == "medium":
            base = 300 + pop * 0.00055
        else:
            base = 700 + pop * 0.00065

        yearly_growth = 0.03
        daily_growth = (1 + yearly_growth) ** (1/365) - 1

        for i, d in enumerate(dates):
            days_from_start = (d - start).days
            trend = base * ((1 + daily_growth) ** days_from_start)

            dow = d.weekday()
            week_season = 1.25 if dow >= 5 else 0.95

            # monthly bump (e.g., آذر ~ Dec)
            month_season = 1.3 if d.month == 12 else (1.15 if d.month == 11 else 1.0)

            promo = 1 if d in promo_calendar[s] else 0
            promo_effect = 1.35 if promo else 1.0

            # holiday spike small probability in Nov/Dec
            holiday_spike = 1.4 if (np.random.rand() < 0.005 and d.month in [11,12]) else 1.0

            comp_effect = 1.0
            if s == "store_2" and d >= competitor_open_date:
                comp_effect = 0.85

            noise = np.random.normal(loc=1.0, scale=0.08)

            sales = trend * week_season * month_season * promo_effect * holiday_spike * comp_effect * noise
            sales = max(0, round(sales, 2))

            rows.append({
                "date": d,
                "store_id": s,
                "daily_sales": sales,
                "promotion_active": int(promo),
                "day_of_week": d.weekday(),
                "month": d.month,
                "is_weekend": int(d.weekday() >= 5),
                "store_size": size,
                "city_population": pop,
            })

    df = pd.DataFrame(rows).sort_values(["store_id", "date"]).reset_index(drop=True)
    return df, competitor_open_date

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df, comp_date = simulate_innovatemart(start_date="2022-01-01", periods=730, n_stores=4)
    df.to_csv("data/simulated_sales.csv", index=False)
    print("Saved to data/simulated_sales.csv")
    print("Competitor opened near store_2 on:", comp_date.date())
