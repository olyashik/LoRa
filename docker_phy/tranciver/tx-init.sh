#!/bin/bash
set -uo pipefail

ip addr add 192.168.1.3/24 dev eth0 2>/dev/null || true
ip link set eth0 up

echo "[TX-init] Интерфейс настроен:"
ip addr show eth0

echo "[TX-init] Проверка связи с RX..."
ping -c 2 "${RX_HOST:-192.168.1.2}" || echo "[TX-init] RX пока не отвечает (возможно ещё не поднялся) — продолжаю"

echo "[TX-init] Отправка пакета..."
python -u tx_main.py

echo "[TX-init] Готово. Оболочка остаётся открытой для консоли GNS3."
echo "[TX-init] Повторный запуск: python -u tx_main.py"
exec /bin/bash