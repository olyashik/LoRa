#!/bin/bash
# Устанавливаем IP
ip addr add 192.168.1.3/24 dev eth0
ip link set eth0 up

# Проверяем связь с RX
ping -c 2 192.168.1.2

# Запускаем передатчик
RX_HOST=192.168.1.2 python -u tx_main.py