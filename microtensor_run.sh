#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 12 0 10 4 15 > System_12/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 12 1 12 4 18 > System_12/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 12 2 15 4 23 > System_12/MyRunInfo/Run_2.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
