#!/bin/bash
#
source ~/miniconda3/bin/activate footprint
sudo systemctl start mosquitto
python mqtt_data2groundmark_simulator.py
