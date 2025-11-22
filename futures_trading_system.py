"""
Futures Trading System - PDF Methodology v3.0
基於多模組 AI 與專家系統的期貨買進信號示例。
"""

import datetime
import os
import subprocess
import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 字體設定
# ---------------------------------------------------------------------------

def configure_chinese_font() -> str:
    """Configure a Chinese-friendly font and return the chosen family name."""
    subprocess.run(["apt-get", "update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    subprocess.run(
        ["apt-get", "install", "-y", "fonts-wqy-microhei"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    cache_dir = os.path.expanduser("~/.matplotlib")
    for cache_file in [os.path.join(cache_dir, "fontList.json"), os.path.join(cache_dir, "fontList.cache")]:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass

    font_path_wqy = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if os.path.exists(font_path_wqy):
        try:
            fm.fontManager.addfont(font_path_wqy)
        except Exception:
            pass

    font_options = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
    selected_font = None
    for font in font_options:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams["font.sans-serif"] = [font]
            selected_font = font
            break

    if not selected_font:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        selected_font = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    return selected_font


# ---------------------------------------------------------------------------
# 技術指標
# ---------------------------------------------------------------------------

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(
    data: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    low_min = data[low_col].rolling(window=k_window).min()
    high_max = data[high_col].rolling(window=k_window).max()
    data["K"] = ((data[close_col] - low_min) / (high_max - low_min)) * 100
    data["D"] = data["K"].rolling(window=d_window).mean()
    return data


def calculate_obv(data: pd.DataFrame, close_col: str = "close", volume_col: str = "volume") -> pd.DataFrame:
    obv: List[int] = [0]
    for i in range(1, len(data)):
        if data[close_col].iloc[i] > data[close_col].iloc[i - 1]:
            obv.append(obv[-1] + data[volume_col].iloc[i])
        elif data[close_col].iloc[i] < data[close_col].iloc[i - 1]:
            obv.append(obv[-1] - data[volume_col].iloc[i])
        else:
            obv.append(obv[-1])

    data["OBV"] = obv
    data["OBV_MA_20"] = data["OBV"].rolling(window=20).mean()
    return data


def calculate_rolling_max_dd(returns: pd.Series, window: int = 20) -> pd.Series:
    max_dd = []
    for i in range(len(returns)):
        subset = returns.iloc[: i + 1] if i < window else returns.iloc[i - window + 1 : i + 1]
        cumulative = (1 + subset).cumprod()
        running_max = cumulative.expanding().max()
        dd = (cumulative - running_max) / running_max
        max_dd.append(dd.min())
    return pd.Series(max_dd, index=returns.index)


# ---------------------------------------------------------------------------
# 評估
# ---------------------------------------------------------------------------

def evaluate_signal_performance(data: pd.DataFrame, confidence_threshold: float = 0.70) -> Optional[Dict[str, float]]:
    signals_df = data[data["buy_signal_strength"] >= confidence_threshold].copy()
    if len(signals_df) == 0:
        return None

    returns = []
    for idx in signals_df.index:
        price_at_signal = data.loc[idx, "close"]
        price_in_20d = data.loc[min(idx + 20, len(data) - 1), "close"]
        returns.append((price_in_20d - price_at_signal) / price_at_signal)

    signal_returns = pd.Series(returns, index=signals_df.index)
    win_count = (signal_returns > 0).sum()
    total_trades = len(signal_returns)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    std_return = signal_returns.std()
    sharpe_ratio = signal_returns.mean() / std_return * np.sqrt(252) if std_return > 0 else 0
    profit_factor = (
        signal_returns[signal_returns > 0].sum() / abs(signal_returns[signal_returns <= 0].sum())
        if (signal_returns <= 0).sum() > 0
        else np.inf
    )

    return {
        "confidence_threshold": confidence_threshold,
        "total_signals": total_trades,
        "win_count": win_count,
        "loss_count": (signal_returns <= 0).sum(),
        "win_rate": win_rate,
        "avg_return": signal_returns.mean(),
        "std_return": std_return,
        "sharpe_ratio": sharpe_ratio,
        "max_profit": signal_returns.max(),
        "max_loss": signal_returns.min(),
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("期貨交易系統 - PDF方法論版本 v3.0")
    print("Futures Trading System - PDF Multi-Module Methodology v3.0")
    print("=" * 80 + "\n")

    print("正在配置中文字體...\n")
    selected_font = configure_chinese_font()
    print(f"使用字體: {selected_font}\n")

    print("=" * 80)
    print("第 1 步：下載期貨數據")
    print("=" * 80 + "\n")

    futures_symbols = {"2330": "2330.TW", "ES": "ES=F", "GC": "GC=F"}
    selected_future = "2330"
    futures_symbol = futures_symbols[selected_future]

    print(f"選擇的期貨: {selected_future} ({futures_symbol})")
    print("下載時間: 2020-01-01 至今\n")

    subprocess.run(["pip", "install", "arch", "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    try:
        print("📍 下載期貨數據...")
        futures_data = yf.download(futures_symbol, start="2020-01-01", progress=False)
        if len(futures_data) == 0:
            raise ValueError("No data downloaded")

        futures_data.index = pd.to_datetime(futures_data.index)
        start_date = str(futures_data.index[0].date())
        end_date = str(futures_data.index[-1].date())
        print(f"✅ 下載成功: {len(futures_data)} 個交易日")
        print(f" 日期範圍: {start_date} 到 {end_date}")
        print(f" 價格範圍: ${futures_data['Close'].min():.2f} - ${futures_data['Close'].max():.2f}\n")
        data_source = "Real Yahoo Finance"
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 下載失敗: {exc}")
        print("📊 使用模擬期貨數據...\n")
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")
        prices = [1000]
        for _ in range(1, len(dates)):
            change = np.random.normal(0.0003, 0.01)
            prices.append(prices[-1] * (1 + change))

        futures_data = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "Close": prices,
                "Adj Close": prices,
                "Volume": np.random.randint(1_000_000, 5_000_000, len(dates)),
            },
            index=dates,
        )
        data_source = "Simulated"
        print(f"✅ 生成模擬數據: {len(futures_data)} 個交易日\n")

    futures_data.columns = futures_data.columns.get_level_values(0).str.lower()

    # 技術指標
    print("=" * 80)
    print("第 2 步：計算技術指標和風險指標")
    print("=" * 80 + "\n")

    print("📈 計算 RSI...")
    futures_data["RSI"] = calculate_rsi(futures_data["close"])

    print("📈 計算 MACD...")
    futures_data["EMA_12"] = futures_data["close"].ewm(span=12, adjust=False).mean()
    futures_data["EMA_26"] = futures_data["close"].ewm(span=26, adjust=False).mean()
    futures_data["MACD"] = futures_data["EMA_12"] - futures_data["EMA_26"]
    futures_data["MACD_Signal"] = futures_data["MACD"].ewm(span=9, adjust=False).mean()
    futures_data["MACD_Histogram"] = futures_data["MACD"] - futures_data["MACD_Signal"]

    print("📈 計算 KD 隨機指標...")
    futures_data = calculate_stochastic(futures_data)

    print("📈 計算 KD 穿越信號...")
    futures_data["K_above_D"] = futures_data["K"] > futures_data["D"]
    futures_data["K_crossover_D"] = futures_data["K_above_D"] & (~futures_data["K_above_D"].shift(1).fillna(False))
    futures_data["K_crossbelow_D"] = (~futures_data["K_above_D"]) & (futures_data["K_above_D"].shift(1).fillna(False))

    print("📈 計算 OBV...")
    futures_data = calculate_obv(futures_data)

    print("📈 計算布林帶...")
    futures_data["BB_Middle"] = futures_data["close"].rolling(window=20).mean()
    bb_std = futures_data["close"].rolling(window=20).std()
    futures_data["BB_Upper"] = futures_data["BB_Middle"] + (bb_std * 2)
    futures_data["BB_Lower"] = futures_data["BB_Middle"] - (bb_std * 2)
    futures_data["BB_Position"] = (futures_data["close"] - futures_data["BB_Lower"]) / (
        futures_data["BB_Upper"] - futures_data["BB_Lower"]
    )

    print("📈 計算移動平均線...")
    futures_data["MA_5"] = futures_data["close"].rolling(window=5).mean()
    futures_data["MA_20"] = futures_data["close"].rolling(window=20).mean()
    futures_data["MA_50"] = futures_data["close"].rolling(window=50).mean()
    futures_data["MA_200"] = futures_data["close"].rolling(window=200).mean()

    print("📈 計算收益率和波動率...")
    futures_data["daily_return"] = futures_data["close"].pct_change()
    futures_data["volatility"] = futures_data["daily_return"].rolling(window=20).std()
    futures_data["volume_ma"] = futures_data["volume"].rolling(window=20).mean()
    futures_data["volume_ratio"] = futures_data["volume"] / futures_data["volume_ma"]

    print("📈 計算 ATR...")
    futures_data["tr"] = np.maximum(
        futures_data["high"] - futures_data["low"],
        np.maximum(
            abs(futures_data["high"] - futures_data["close"].shift(1)),
            abs(futures_data["low"] - futures_data["close"].shift(1)),
        ),
    )
    futures_data["ATR"] = futures_data["tr"].rolling(window=14).mean()

    print("📈 計算 Sharpe Ratio...")
    risk_free_rate = 0.02 / 252
    futures_data["sharpe_20d"] = (
        (futures_data["daily_return"].rolling(window=20).mean() - risk_free_rate)
        / (futures_data["daily_return"].rolling(window=20).std() + 1e-6)
        * np.sqrt(252)
    )

    print("📈 計算 Max Drawdown...")
    futures_data["max_dd_20d"] = calculate_rolling_max_dd(futures_data["daily_return"])
    print("✅ 所有指標計算完成\n")

    # 買進信號
    print("=" * 80)
    print("第 3 步：多模組買進信號系統（PDF方法論）")
    print("=" * 80 + "\n")

    futures_data["buy_signal_strength"] = 0.0
    futures_data["buy_modules"] = [[] for _ in range(len(futures_data))]

    module_1 = (
        (futures_data["RSI"] < 30)
        & (futures_data["MACD"] > futures_data["MACD_Signal"])
        & (futures_data["K"] > futures_data["D"])
    )
    futures_data.loc[module_1, "buy_signal_strength"] = np.maximum(
        futures_data.loc[module_1, "buy_signal_strength"], 0.8
    )
    for idx in module_1[module_1].index:
        futures_data.loc[idx, "buy_modules"].append("Module-1: 超賣反彈")
    print(f"📌 模組 1: 超賣反彈模組 -> 觸發 {module_1.sum()} 次\n")

    module_2 = (
        (futures_data["K"] < 5)
        & (futures_data["RSI"] < 25)
        & (futures_data["close"] > futures_data["BB_Lower"])
    )
    futures_data.loc[module_2, "buy_signal_strength"] = np.maximum(
        futures_data.loc[module_2, "buy_signal_strength"], 0.75
    )
    for idx in module_2[module_2].index:
        futures_data.loc[idx, "buy_modules"].append("Module-2: K極度超賣")
    print(f"📌 模組 2: K線極度超賣模組 -> 觸發 {module_2.sum()} 次\n")

    module_3 = (
        (futures_data["K_crossover_D"])
        & (futures_data["MACD"] > futures_data["MACD_Signal"])
        & (futures_data["OBV"] > futures_data["OBV_MA_20"])
    )
    futures_data.loc[module_3, "buy_signal_strength"] = np.maximum(
        futures_data.loc[module_3, "buy_signal_strength"], 0.85
    )
    for idx in module_3[module_3].index:
        futures_data.loc[idx, "buy_modules"].append("Module-3: 雙指標確認")
    print(f"📌 模組 3: 雙指標確認模組 -> 觸發 {module_3.sum()} 次\n")

    module_4 = (
        (futures_data["close"] > futures_data["MA_20"])
        & (futures_data["MA_20"] > futures_data["MA_50"])
        & (futures_data["MACD"] > futures_data["MACD_Signal"])
        & (futures_data["K"] > 50)
    )
    futures_data.loc[module_4, "buy_signal_strength"] = np.maximum(
        futures_data.loc[module_4, "buy_signal_strength"], 0.70
    )
    for idx in module_4[module_4].index:
        futures_data.loc[idx, "buy_modules"].append("Module-4: 趨勢追蹤")
    print(f"📌 模組 4: 趨勢追蹤模組 -> 觸發 {module_4.sum()} 次\n")

    module_5 = (
        (futures_data["sharpe_20d"] > 1.0)
        & (futures_data["max_dd_20d"] > -0.15)
        & (futures_data["RSI"] < 40)
        & (futures_data["MACD"] > futures_data["MACD_Signal"])
    )
    futures_data.loc[module_5, "buy_signal_strength"] = np.maximum(
        futures_data.loc[module_5, "buy_signal_strength"], 0.72
    )
    for idx in module_5[module_5].index:
        futures_data.loc[idx, "buy_modules"].append("Module-5: 風險調整")
    print(f"📌 模組 5: 風險調整模組 -> 觸發 {module_5.sum()} 次\n")

    buy_signals = futures_data[futures_data["buy_signal_strength"] > 0.0]
    high_conf_signals = futures_data[futures_data["buy_signal_strength"] >= 0.80]
    medium_conf_signals = futures_data[
        (futures_data["buy_signal_strength"] >= 0.70) & (futures_data["buy_signal_strength"] < 0.80)
    ]
    low_conf_signals = futures_data[
        (futures_data["buy_signal_strength"] > 0.0) & (futures_data["buy_signal_strength"] < 0.70)
    ]

    print("📊 買進信號統計（PDF 多模組方法論）：")
    print(f"✅ 高置信度信號 (≥0.80): {len(high_conf_signals)} 次")
    print(f"✅ 中置信度信號 (0.70-0.80): {len(medium_conf_signals)} 次")
    print(f"✅ 低置信度信號 (0.0-0.70): {len(low_conf_signals)} 次")
    print(f"✅ 總買進信號: {len(buy_signals)} 次\n")

    # 機器學習驗證
    print("=" * 80)
    print("第 4 步：機器學習 - Random Forest 驗證")
    print("=" * 80 + "\n")

    features = [
        "RSI",
        "MACD",
        "MACD_Histogram",
        "MA_20",
        "MA_50",
        "MA_200",
        "volatility",
        "volume_ratio",
        "BB_Position",
        "ATR",
        "K",
        "D",
        "OBV",
        "OBV_MA_20",
        "sharpe_20d",
        "max_dd_20d",
    ]

    futures_data["future_return_20d"] = futures_data["close"].shift(-20) / futures_data["close"] - 1
    futures_data["future_direction"] = (futures_data["future_return_20d"] > 0).astype(int)

    X = futures_data[features].copy()
    y = futures_data["future_direction"].copy()
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"訓練數據: {len(X)} 個樣本\n")

    if len(X) > 100:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )

        print(f"訓練集: {len(X_train)}, 測試集: {len(X_test)}\n")

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        y_pred_test = rf_model.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred_test) * 100
        test_precision = precision_score(y_test, y_pred_test, zero_division=0) * 100
        test_recall = recall_score(y_test, y_pred_test, zero_division=0) * 100

        print(f"測試集準確度: {test_accuracy:.2f}%")
        print(f"測試集精準度: {test_precision:.2f}%")
        print(f"測試集召回率: {test_recall:.2f}%\n")

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(rf_model, X_scaled, y, cv=tscv)
        cv_mean = cv_scores.mean() * 100
        cv_std = cv_scores.std() * 100

        print(f"交叉驗證準確度: {cv_mean:.2f}% ± {cv_std:.2f}%\n")

        feature_importance = pd.DataFrame(
            {"feature": features, "importance": rf_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("特徵重要性 Top 10:")
        print(feature_importance.head(10).to_string(index=False))
        print()
    else:
        print("樣本數不足，跳過機器學習評估。\n")

    # 當前市場狀態
    print("\n" + "=" * 80)
    print("第 5 步：當前市場狀態分析")
    print("=" * 80 + "\n")

    current_price = futures_data["close"].iloc[-1]
    current_rsi = futures_data["RSI"].iloc[-1]
    current_k = futures_data["K"].iloc[-1]
    current_d = futures_data["D"].iloc[-1]
    current_macd = futures_data["MACD"].iloc[-1]
    current_signal = futures_data["MACD_Signal"].iloc[-1]
    current_buy_strength = futures_data["buy_signal_strength"].iloc[-1]
    current_sharpe = futures_data["sharpe_20d"].iloc[-1]
    current_max_dd = futures_data["max_dd_20d"].iloc[-1]

    print("📊 當前市場狀態:")
    print(f" 價格: ${current_price:.2f}")
    print(f" RSI: {current_rsi:.2f}")
    print(f" K 線: {current_k:.2f}")
    print(f" D 線: {current_d:.2f}")
    print(f" K > D: {'✓ 是' if current_k > current_d else '✗ 否'}")
    print(f" MACD: {current_macd:.6f}")
    print(f" Signal: {current_signal:.6f}")
    print(f" MACD > Sig: {'✓ 是' if current_macd > current_signal else '✗ 否'}")
    print(f" Sharpe (20d): {current_sharpe:.4f}")
    print(f" Max DD (20d): {current_max_dd:.4f}")
    print(f"\n🎯 買進信號強度: {current_buy_strength:.2%}")
    modules_display = futures_data["buy_modules"].iloc[-1] if futures_data["buy_modules"].iloc[-1] else "無"
    print(f"✅ 當前觸發模組: {modules_display}\n")

    # 績效評估
    print("=" * 80)
    print("第 6 步：績效評估（按 PDF 多模組標準）")
    print("=" * 80 + "\n")

    for threshold in [0.70, 0.75, 0.80]:
        performance = evaluate_signal_performance(futures_data, threshold)
        if performance:
            print(f"置信度閾值 ≥ {threshold}:")
            print(f" 信號數量: {performance['total_signals']}")
            print(f" 勝率: {performance['win_rate']:.2%}")
            print(f" 平均收益: {performance['avg_return']:.2%}")
            print(f" Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
            print(f" 最大單筆利潤: {performance['max_profit']:.2%}")
            print(f" 最大單筆虧損: {performance['max_loss']:.2%}")
            print(f" 獲利因子: {performance['profit_factor']:.2f}\n")

    print("=" * 80)
    print("✅ v3.0 多模組 PDF 方法論版本完成！")
    print("資料來源:", data_source)
    print("=" * 80)


if __name__ == "__main__":
    main()
