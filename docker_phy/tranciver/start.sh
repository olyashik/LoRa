#!/bin/bash
ip addr add 192.168.100.3/24 dev eth0 2>/dev/null
ip link set eth0 up
sleep 2
python tx_main.py