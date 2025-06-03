FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY data/ ./data
ENV PYTHONPATH=/app/src

# ここだけ変更：標準入出力サービス起動
CMD ["python", "-u", "src/route_cli.py"]
