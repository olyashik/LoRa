#!/bin/bash
set -uo pipefail

# Настраиваем интерфейс (игнорируем ошибку, если адрес уже назначен)
ip addr add 192.168.1.2/24 dev eth0 2>/dev/null || true
ip link set eth0 up

echo "[RX-init] Интерфейс настроен:"
ip addr show eth0

echo "[RX-init] Запуск приёмника, слушаю UDP 5005..."
exec python -u rx_main.py