#!/bin/bash
# Устанавливаем IP
ip addr add 192.168.1.2/24 dev eth0
ip link set eth0 up

# Проверяем сеть
ip addr show eth0
echo "RX готов, слушает UDP порт 5005"

# Запускаем приемник
python -u rx_main.py