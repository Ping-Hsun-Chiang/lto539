from src.backtest.engine import summarise, random_baseline


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

    notable = [r for r in results if r["hits"] >= 3]
    if notable:
        print(f"\n  Notable (3+ hits): {len(notable)} times")
        for r in notable[-5:]:
            print(f"    {r['date']} ({r['weekday'][:3]}) "
                  f"pred={r['predicted']} actual={r['actual']} hits={r['hits']}")
    print()


def print_comparison(all_results: dict[str, list[dict] | dict]) -> None:
    """
    all_results: {label: results_list_or_summary_dict}
    Special label 'Random' maps to a pre-computed summary dict.
    """
    # build summaries
    summaries = {}
    total = None
    for label, val in all_results.items():
        if label == "Random":
            summaries[label] = val
        else:
            summaries[label] = summarise(val)
            total = total or len(val)

    labels = list(summaries.keys())
    col_w = 10

    header = f"  {'Hits':<6}" + "".join(f"{l:>{col_w}}" for l in labels)
    sep = "  " + "-" * (6 + col_w * len(labels))

    print("\n" + "=" * len(sep))
    print("  6-Way Comparison — Hit Distribution (%)")
    if total:
        print(f"  Predictions per method: {total}")
    print("=" * len(sep))
    print(header)
    print(sep)

    for hits in range(6):
        row = f"  {hits} hit(s)"
        for label in labels:
            pct = summaries[label][hits]["pct"]
            row += f"{pct:>{col_w}.1f}%"
        print(row)

    print(sep)

    # highlight best method per row (excluding Random)
    print("\n  Best model per hit count (excluding Random):")
    real_labels = [l for l in labels if l != "Random"]
    for hits in range(1, 6):
        best = max(real_labels, key=lambda l: summaries[l][hits]["pct"])
        pct = summaries[best][hits]["pct"]
        if pct > 0:
            print(f"    {hits} hit(s): {best} ({pct:.1f}%)")
    print()
