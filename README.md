# lto539 — 今彩539 深度學習預測系統

結合傳統抓牌邏輯與深度學習，對台灣彩券「今彩539」進行 Walk-Forward 回測與號碼預測。

## 功能

- 自動從台灣彩券官方 API 抓取歷史開獎資料
- 9 種傳統抓牌特徵工程（熱號、遺漏值、號頭、連號…）
- LSTM + Rule-based Ensemble 預測下期 5 個號碼
- Walk-Forward 回測：每期用前期資料訓練，統計命中 0~5 顆分布

## CLI 指令

```bash
python cli/main.py collect          # 下載/更新歷史資料
python cli/main.py train            # 訓練模型
python cli/main.py predict          # 預測下一期 5 顆號碼
python cli/main.py backtest         # Walk-Forward 回測，輸出命中分布
```

## 資料來源

台灣彩券官方 API：`https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Daily539Result`

## 聲明

此專案純屬統計實驗，樂透為獨立機率事件，無法真正預測，請勿用於實際投注決策。
