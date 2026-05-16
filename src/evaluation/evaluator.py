from src.backtest.engine import summarise


def print_report(results: list[dict]) -> None:
    summary = summarise(results)
    total = len(results)

    print("\n" + "=" * 50)
    print("  Walk-Forward Backtest — Hit Distribution")
    print(f"  Total predictions: {total}")
    print("=" * 50)

    bar_max = 30
    max_pct = max(v["pct"] for v in summary.values()) or 1

    for hits in range(6):
        count = summary[hits]["count"]
        pct = summary[hits]["pct"]
        bar_len = int(bar_max * pct / max_pct)
        bar = "#" * bar_len
        print(f"  {hits} hit(s): {bar:<{bar_max}} {count:>5} draws ({pct:5.1f}%)")

    print("=" * 50)

    # highlight any 3+ hit occurrences
    notable = [r for r in results if r["hits"] >= 3]
    if notable:
        print(f"\n  Notable (3+ hits): {len(notable)} times")
        for r in notable[-5:]:  # show last 5
            print(f"    {r['date']} ({r['weekday'][:3]}) "
                  f"pred={r['predicted']} actual={r['actual']} hits={r['hits']}")
    print()
