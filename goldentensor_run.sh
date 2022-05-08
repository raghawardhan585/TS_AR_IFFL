#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 15 3 15 4 30 > System_15/MyRunInfo/Run_3.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
