import csv
import os
import time
import warnings
from datetime import datetime, timedelta

import requests
import urllib3

# Taiwan Lottery API certificate missing Subject Key Identifier (Python 3.14 strict SSL)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Daily539Result"
RAW_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/history.csv")
FIELDNAMES = ["period", "date", "weekday", "n1", "n2", "n3", "n4", "n5"]


def _fetch_month(year: int, month: int) -> list[dict]:
    url = f"{BASE_URL}?period&month={year}-{month:02d}&pageSize=31"
    try:
        resp = requests.get(url, timeout=10, verify=False)
        resp.raise_for_status()
        data = resp.json()
        results = data["content"]["daily539Res"]
    except Exception as e:
        print(f"  [WARN] Failed to fetch {year}-{month:02d}: {e}")
        return []

    rows = []
    for r in results:
        dt = datetime.fromisoformat(r["lotteryDate"])
        nums = sorted(r["drawNumberSize"])
        rows.append({
            "period": r["period"],
            "date": dt.strftime("%Y-%m-%d"),
            "weekday": dt.strftime("%A"),
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4],
        })
    return rows


def collect(years: int = 2, output_path: str = RAW_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    today = datetime.today()
    start = today - timedelta(days=365 * years)
    all_rows = []

    cur = start.replace(day=1)
    while cur <= today:
        print(f"  Fetching {cur.year}-{cur.month:02d} ...", end=" ", flush=True)
        rows = _fetch_month(cur.year, cur.month)
        print(f"{len(rows)} draws")
        all_rows.extend(rows)
        # advance one month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
        time.sleep(0.3)

    # deduplicate and sort by period ascending
    seen = set()
    unique = []
    for r in all_rows:
        if r["period"] not in seen:
            seen.add(r["period"])
            unique.append(r)
    unique.sort(key=lambda x: x["period"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(unique)

    print(f"\n[OK] Saved {len(unique)} draws to {output_path}")
    return output_path


def load(path: str = RAW_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}. Run: python cli/main.py collect")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "period": int(row["period"]),
                "date": row["date"],
                "weekday": row["weekday"],
                "numbers": [int(row[f"n{i}"]) for i in range(1, 6)],
            })
    return rows
