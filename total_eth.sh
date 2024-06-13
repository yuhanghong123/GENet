#!/bin/bash

# 运行 Python 脚本 total_eth.py 并传递参数
python trainer.py -c config/train/config_eth.yaml 
python total_eth.py -s config/train/config_eth.yaml -t config/test/config_diap.yaml
python total_eth.py -s config/train/config_eth.yaml -t config/test/config_mpii.yaml

