# lto539 — 今彩539 深度學習預測系統

結合傳統抓牌邏輯與深度學習，對台灣彩券「今彩539」進行 Walk-Forward 回測與號碼預測。

🌐 **線上網站**：https://lto539-bdcx4jzzfkiipgaouhyhha.streamlit.app/

---

## 功能

- 自動從台灣彩券官方 API 抓取歷史開獎資料（每日自動更新）
- 9 種傳統抓牌特徵工程
- Rule-based + LSTM Ensemble 預測下期 5 個號碼
- Walk-Forward 回測：6 種策略對比（累積視窗、滾動 6/12/18/24 期、隨機基準）
- Streamlit 網頁介面：預測熱度圖、回測比較、特徵分析、歷史查詢

## 傳統抓牌特徵

| 特徵 | 說明 |
|------|------|
| 熱號/冷號 | 近 6/12/18/24 期出現頻率 |
| 遺漏值 | 距上次出現已過幾期 |
| 平均週期 | 該號碼平均多少期出現一次 |
| 共現矩陣 | 哪些號碼常同期出現 |
| 條件機率 | 上期出現 A，本期 B 跟著出現的機率 |
| 星期規律 | 同星期幾的號碼分佈偏好 |
| 連號分析 | 連續號碼出現頻率 |
| 尾數分佈 | 個位數 0-9 的出現比例 |
| 號頭分析 | 0頭(1-9)、1頭(10-19)、2頭(20-29)、3頭(30-39) |

## 回測結果（616 期，2024-06 ～ 2026-05）

| 命中顆數 | Expanding | Rolling-6 | Rolling-12 | Rolling-18 | Rolling-24 | Random |
|---------|-----------|-----------|------------|------------|------------|--------|
| 0 顆 | 47.1% | 45.9% | 47.7% | 46.6% | 48.4% | 48.3% |
| 1 顆 | 42.0% | 43.5% | 40.4% | 41.7% | 39.3% | 40.3% |
| 2 顆 | 9.6% | 9.7% | 11.0% | 10.6% | 10.7% | 10.4% |
| 3 顆 | 1.3% | 0.8% | 0.8% | 1.0% | 1.3% | 1.0% |
| 4 顆 | 0.0% | 0.0% | 0.0% | 0.2% | 0.3% | 0.0% |
| 5 顆 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## CLI 指令

```bash
# 下載/更新歷史資料
python cli/main.py collect

# 訓練 LSTM 模型（本機用）
python cli/main.py train

# 預測下一期 5 顆號碼
python cli/main.py predict

# Walk-Forward 回測（累積視窗）
python cli/main.py backtest

# 六方法完整比較
python cli/main.py compare

# 啟動網頁介面
python -m streamlit run web/app.py
```

## 專案結構

```
lto539/
├── src/
│   ├── data/           # 資料收集與載入
│   ├── features/       # 9 種傳統抓牌特徵
│   ├── models/         # Rule-based、LSTM、Ensemble
│   ├── backtest/       # Walk-Forward 回測引擎
│   └── evaluation/     # 命中統計與報告
├── cli/main.py         # 命令列介面
├── web/app.py          # Streamlit 網頁介面
├── data/raw/           # 歷史開獎資料（自動更新）
└── .github/workflows/  # 每日自動更新排程
```

## 自動更新

GitHub Actions 每天 23:30（台灣時間，週一至週六）自動執行：
1. 抓取最新開獎資料
2. Push 到 GitHub
3. Streamlit Cloud 自動重新部署

## 資料來源

台灣彩券官方 API：`https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Daily539Result`

## 聲明

此專案純屬統計實驗與技術研究，樂透為獨立機率事件，任何模型皆無法真正預測開獎結果，請勿用於實際投注決策。
