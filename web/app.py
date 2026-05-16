import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.backtest.engine import random_baseline, summarise
from src.data.collector import load
from src.features.traditional import (
    avg_cycle, co_occurrence, conditional_prob, consecutive,
    head_digit, hot_cold, missing_value, tail_digit, weekday_pattern,
)
from src.models.ensemble import EnsembleModel

# ── constants ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(__file__))
FEATURE_FNS = [
    hot_cold, missing_value, avg_cycle, co_occurrence,
    conditional_prob, weekday_pattern, consecutive, tail_digit, head_digit,
]
FEATURE_NAMES = [
    "熱號/冷號", "遺漏值", "平均週期", "共現矩陣",
    "條件機率", "星期規律", "連號分析", "尾數分佈", "號頭分析",
]
ROLLING_WINDOWS = [6, 12, 18, 24]
NUMBERS = list(range(1, 40))


# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="載入歷史資料...")
def get_history():
    return load(os.path.join(ROOT, "data/raw/history.csv"))


@st.cache_data(show_spinner="計算預測分數...")
def get_scores(last_period: int):
    history = get_history()
    model = EnsembleModel(use_lstm=False)  # rule-based only for web (no torch needed)
    model.fit(history)
    scores = model.predict_proba(history)
    top5 = model.top5(history)
    return scores, top5


@st.cache_data(show_spinner="載入回測結果...")
def get_backtest_results():
    results = {}
    # expanding
    path = os.path.join(ROOT, "backtest_results.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            results["Expanding"] = json.load(f)
    # rolling
    for w in ROLLING_WINDOWS:
        path = os.path.join(ROOT, f"backtest_results_rolling_{w}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                results[f"Rolling-{w}"] = json.load(f)
    return results


# ── helpers ────────────────────────────────────────────────────────────────────
def make_number_grid(scores: np.ndarray, top5: list[int]) -> go.Figure:
    """4×10 heatmap grid for numbers 1-39."""
    rows, cols = 4, 10
    z = np.full((rows, cols), np.nan)
    text = [[""] * cols for _ in range(rows)]
    for n in NUMBERS:
        r, c = (n - 1) // cols, (n - 1) % cols
        z[r][c] = float(scores[n - 1])
        text[r][c] = str(n)

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="Score"),
        hovertemplate="號碼 %{text}<br>分數: %{z:.4f}<extra></extra>",
    ))

    # highlight top5 with border
    for n in top5:
        r, c = (n - 1) // cols, (n - 1) % cols
        fig.add_shape(type="rect",
            x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
            line=dict(color="blue", width=3))

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
    )
    return fig


def make_comparison_chart(all_summaries: dict) -> go.Figure:
    labels = list(all_summaries.keys())
    colors = ["#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#4CAF50", "#9E9E9E"]
    fig = go.Figure()
    for i, label in enumerate(labels):
        pcts = [all_summaries[label][k]["pct"] for k in range(6)]
        fig.add_trace(go.Bar(
            name=label,
            x=[f"{k} 顆" for k in range(6)],
            y=pcts,
            marker_color=colors[i % len(colors)],
            text=[f"{p:.1f}%" for p in pcts],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        height=420,
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(title="比例 (%)"),
        xaxis=dict(title="命中顆數"),
        margin=dict(t=20, b=60),
    )
    return fig


def make_feature_chart(scores: np.ndarray, feature_name: str, top5: list[int]) -> go.Figure:
    colors = ["#E53935" if n in top5 else "#1976D2" for n in NUMBERS]
    fig = go.Figure(go.Bar(
        x=[str(n) for n in NUMBERS],
        y=scores.tolist(),
        marker_color=colors,
        hovertemplate="號碼 %{x}<br>分數: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=feature_name,
        height=320,
        xaxis=dict(title="號碼", tickmode="linear"),
        yaxis=dict(title="分數"),
        margin=dict(t=40, b=40),
    )
    return fig


# ── page layout ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="今彩539 預測系統", page_icon="🎲", layout="wide")
st.title("🎲 今彩539 深度學習預測系統")

history = get_history()
last = history[-1]
scores, top5 = get_scores(last["period"])

tab1, tab2, tab3, tab4 = st.tabs(["🎯 預測", "📊 回測比較", "🔍 特徵分析", "📋 歷史資料"])


# ══ Tab 1: 預測 ══════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("上期開獎")
        st.markdown(f"**期號：** {last['period']}")
        st.markdown(f"**日期：** {last['date']} ({last['weekday'][:3]})")
        nums_str = "　".join(f"**{n}**" for n in last["numbers"])
        st.markdown(f"**號碼：** {nums_str}")

        st.divider()
        st.subheader("下期預測號碼")
        top5_str = "　".join(f"🔵 **{n}**" for n in top5)
        st.markdown(top5_str)
        st.caption("藍框 = 預測號碼；顏色越深 = 分數越高")

    with col_r:
        st.subheader("號碼熱度圖（1–39）")
        st.plotly_chart(make_number_grid(scores, top5), use_container_width=True)

    st.divider()
    st.subheader("各號碼預測分數")
    score_df = pd.DataFrame({
        "號碼": NUMBERS,
        "分數": [round(float(scores[n - 1]), 4) for n in NUMBERS],
        "預測": ["✅" if n in top5 else "" for n in NUMBERS],
    })
    st.dataframe(score_df, use_container_width=True, hide_index=True, height=200)


# ══ Tab 2: 回測比較 ══════════════════════════════════════════════════════════
with tab2:
    bt_results = get_backtest_results()

    if not bt_results:
        st.warning("找不到回測結果檔案，請先執行 `python cli/main.py compare`")
    else:
        # build summaries
        summaries = {}
        n_preds = 616
        for label, res in bt_results.items():
            summaries[label] = summarise(res)
            n_preds = len(res)
        summaries["Random"] = random_baseline(n_preds)

        st.subheader(f"六方法命中分布比較（共 {n_preds} 期預測）")
        st.plotly_chart(make_comparison_chart(summaries), use_container_width=True)

        # table
        st.subheader("詳細數字")
        table_data = {"命中顆數": [f"{k} 顆" for k in range(6)]}
        for label, s in summaries.items():
            table_data[label] = [f"{s[k]['pct']:.1f}%" for k in range(6)]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # notable hits
        all_notable = []
        for label, res in bt_results.items():
            for r in res:
                if r["hits"] >= 3:
                    all_notable.append({
                        "方法": label,
                        "日期": r["date"],
                        "星期": r["weekday"][:3],
                        "預測": str(r["predicted"]),
                        "實際": str(r["actual"]),
                        "命中": r["hits"],
                    })
        if all_notable:
            st.subheader(f"3+ 顆命中紀錄（共 {len(all_notable)} 筆）")
            df_notable = pd.DataFrame(all_notable).sort_values("命中", ascending=False)
            st.dataframe(df_notable, use_container_width=True, hide_index=True)


# ══ Tab 3: 特徵分析 ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("傳統抓牌特徵視覺化")
    st.caption("以目前所有歷史資料計算各號碼在每個特徵下的分數，紅色為本次預測的5顆號碼。")

    selected = st.selectbox("選擇特徵", FEATURE_NAMES)
    fn = FEATURE_FNS[FEATURE_NAMES.index(selected)]

    feat_scores = fn(history)
    st.plotly_chart(make_feature_chart(feat_scores, selected, top5), use_container_width=True)

    # all features summary table
    with st.expander("全部特徵分數總表"):
        feat_df = pd.DataFrame({"號碼": NUMBERS})
        for name, fn in zip(FEATURE_NAMES, FEATURE_FNS):
            s = fn(history)
            feat_df[name] = [round(float(s[n - 1]), 4) for n in NUMBERS]
        feat_df["預測"] = ["✅" if n in top5 else "" for n in NUMBERS]
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ══ Tab 4: 歷史資料 ══════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"歷史開獎紀錄（共 {len(history)} 期）")

    hist_df = pd.DataFrame([{
        "期號": r["period"],
        "日期": r["date"],
        "星期": r["weekday"][:3],
        "號碼": " ".join(f"{n:02d}" for n in r["numbers"]),
        "n1": r["numbers"][0], "n2": r["numbers"][1],
        "n3": r["numbers"][2], "n4": r["numbers"][3], "n5": r["numbers"][4],
    } for r in reversed(history)])

    col_s, col_e = st.columns(2)
    with col_s:
        search_num = st.number_input("搜尋包含號碼", min_value=0, max_value=39, value=0)
    with col_e:
        weekday_filter = st.selectbox(
            "篩選星期",
            ["全部", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        )

    filtered = hist_df.copy()
    if search_num > 0:
        mask = (
            (filtered["n1"] == search_num) | (filtered["n2"] == search_num) |
            (filtered["n3"] == search_num) | (filtered["n4"] == search_num) |
            (filtered["n5"] == search_num)
        )
        filtered = filtered[mask]
    if weekday_filter != "全部":
        filtered = filtered[filtered["星期"] == weekday_filter]

    st.dataframe(
        filtered[["期號", "日期", "星期", "號碼"]],
        use_container_width=True,
        hide_index=True,
        height=500,
    )
    st.caption(f"顯示 {len(filtered)} / {len(hist_df)} 筆")
