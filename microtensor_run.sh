#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 12 3 15 4 30 > System_12/MyRunInfo/Run_3.txt & 
wait
python3 deepDMD.py '/cpu:0' 12 4 20 4 40 > System_12/MyRunInfo/Run_4.txt & 
wait
python3 deepDMD.py '/cpu:0' 12 5 25 4 50 > System_12/MyRunInfo/Run_5.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
