#!/bin/bash

# 运行 Python 脚本 total_gaze360.py 并传递参数
python trainer.py -c config/train/config_gaze360.yaml
python total_gaze360.py -s config/train/config_gaze360.yaml -t config/test/config_gaze360_diap.yaml
python total_gaze360.py -s config/train/config_gaze360.yaml -t config/test/config_gaze360_mpii.yaml

