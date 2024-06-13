import sys
import os
import runpy

os.chdir("/home/hyh1/gaze_domain_adption/code/PureGaze")
args = 'python trainer.py -c config/train/config_gaze360.yaml'

args = args.split()

if args[0] == 'python':
    args.pop(0) 
if args[0] == '-m':
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')
