#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 15 0 15 4 30 > System_15/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 1 20 4 40 > System_15/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 2 25 4 50 > System_15/MyRunInfo/Run_2.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
