#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 51 0 8 4 16 > System_51/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 51 1 12 4 24 > System_51/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 52 0 8 4 16 > System_52/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 52 1 12 4 24 > System_52/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 53 0 8 4 16 > System_53/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 53 1 12 4 24 > System_53/MyRunInfo/Run_1.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
