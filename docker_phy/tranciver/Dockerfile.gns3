FROM python:3.12-slim
WORKDIR /app

# Устанавливаем утилиты для сетевой диагностики
RUN apt-get update && apt-get install -y \
    iproute2 \
    net-tools \
    iputils-ping \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Переменные для UDP
ENV RX_HOST=192.168.1.2
ENV RX_PORT=5005

# Запускаем передатчик
CMD ["python", "-u", "tx_main.py"]