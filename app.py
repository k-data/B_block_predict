
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib
from datetime import datetime, timedelta

# --- タイトル ---
st.title("📈 Bブロック予測アプリ")
st.caption("伝票日付とB在庫を入力するだけで、最大30日分のBブロック予測と95%信頼区間を表示します。")

# --- モデル読み込み ---
@st.cache_resource
def load_model():
    return joblib.load("stacked_best.pkl")

model = load_model()

# --- CSVアップロード ---
uploaded_file = st.file_uploader("📂 管理表CSVをアップロード", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["伝票日付"])
    df = df.sort_values("伝票日付").reset_index(drop=True)
else:
    st.stop()

# --- 特徴量生成 ---
def generate_features(df):
    df = df.copy()
    df["Bブロック_lag1"] = df["Bブロック"].shift(1)
    df["Bブロック_ma3"] = df["Bブロック"].rolling(3).mean()
    df["Bブロック_ma7"] = df["Bブロック"].rolling(7).mean()
    df["Bブロック_pct"] = df["Bブロック"].pct_change()
    df["weekday"] = df["伝票日付"].dt.weekday
    df["B滞留率_lag1"] = df["B在庫"].shift(1) / df["Bブロック"].shift(1)
    df["B滞留率_ma3"] = df["B滞留率_lag1"].rolling(3).mean()
    df["B滞留率_diff"] = df["B滞留率_lag1"] - df["B滞留率_lag1"].shift(1)
    df["Bブロック_ma3_lag1"] = df["Bブロック"].shift(1).rolling(3).mean()
    return df

# --- UI入力 ---
col1, col2 = st.columns(2)
with col1:
    target_date = st.date_input("予測開始日 (伝票日付)", datetime.today())
with col2:
    forecast_days = st.slider("予測日数（最大30日）", 1, 30, 7)

initial_stock = st.number_input("初日のBブロック在庫", min_value=0.0, value=100.0, step=1.0)

# --- 予測実行 ---
if st.button("📊 予測を実行"):
    df_base = df.copy()
    results = []
    current_date = pd.to_datetime(target_date)
    current_stock = initial_stock

    features = [
        "Bブロック_lag1", "Bブロック_ma3", "Bブロック_ma7", "Bブロック_pct",
        "weekday", "B在庫", "B滞留率_lag1", "B滞留率_ma3", "B滞留率_diff", "Bブロック_ma3_lag1"
    ]

    # モデルの残差標準偏差を事前計算
    df_train = generate_features(df_base).dropna()
    X_train = df_train[features]
    y_train = df_train["Bブロック"]
    y_pred = model.predict(X_train)
    std_error = np.std(y_pred - y_train)

    for _ in range(forecast_days):
        df_feat = generate_features(df_base)
        prev_date = current_date - timedelta(days=1)
        row = df_feat[df_feat["伝票日付"] == prev_date]

        if row.empty or row[features].isnull().any(axis=1).values[0]:
            results.append((current_date.date(), None, None, None))
        else:
            input_df = pd.DataFrame([{
                "Bブロック_lag1": row["Bブロック_lag1"].values[0],
                "Bブロック_ma3": row["Bブロック_ma3"].values[0],
                "Bブロック_ma7": row["Bブロック_ma7"].values[0],
                "Bブロック_pct": row["Bブロック_pct"].values[0],
                "weekday": current_date.weekday(),
                "B在庫": current_stock,
                "B滞留率_lag1": row["B滞留率_lag1"].values[0],
                "B滞留率_ma3": row["B滞留率_ma3"].values[0],
                "B滞留率_diff": row["B滞留率_diff"].values[0],
                "Bブロック_ma3_lag1": row["Bブロック_ma3_lag1"].values[0]
            }])
            pred = model.predict(input_df)[0]
            lower = pred - 1.96 * std_error
            upper = pred + 1.96 * std_error
            results.append((current_date.date(), round(pred, 2), round(lower, 2), round(upper, 2)))

            # 仮データを追加
            df_base = pd.concat([df_base, pd.DataFrame({
                "伝票日付": [current_date],
                "Bブロック": [pred],
                "B在庫": [current_stock]
            })], ignore_index=True)

        current_date += timedelta(days=1)

    result_df = pd.DataFrame(results, columns=["予測日", "Bブロック予測", "下限 (95%CI)", "上限 (95%CI)"])
    result_df["予測日"] = pd.to_datetime(result_df["予測日"])

    # 表示
    st.dataframe(result_df.set_index("予測日"))

    # グラフ
    st.subheader("📈 予測グラフ（95%信頼区間付き）")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result_df["予測日"], result_df["Bブロック予測"], marker='o', label="予測値")
    ax.fill_between(result_df["予測日"], result_df["下限 (95%CI)"], result_df["上限 (95%CI)"], color='orange', alpha=0.2, label="95%信頼区間")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_ylabel("Bブロック予測値")
    ax.set_xlabel("日付")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
