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

# Открываем UDP порт
EXPOSE 5005/udp

# Запускаем приемник
CMD ["python", "-u", "rx_main.py"]