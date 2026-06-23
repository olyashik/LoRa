#!/bin/bash
ip addr add 192.168.100.2/24 dev eth0 2>/dev/null
ip link set eth0 up
python rx_main.py