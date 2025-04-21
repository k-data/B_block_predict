# Dockerfile for Streamlit App to run Bブロック予測
FROM python:3.10-slim

# タイムゾーン設定
ENV TZ=Asia/Tokyo

# 作業ディレクトリ作成
WORKDIR /app

# 必要ファイルのコピー
COPY requirements.txt ./
COPY . /app

# 必要パッケージのインストール
RUN pip install --no-cache-dir -r requirements.txt

# ポート開放
EXPOSE 8501

# Streamlit起動コマンド
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
