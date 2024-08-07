#!/bin/bash


python trainer.py -c config/train/config_eth.yaml 
python total_eth.py -s config/train/config_eth.yaml -t config/test/config_diap.yaml
python total_eth.py -s config/train/config_eth.yaml -t config/test/config_mpii.yaml

