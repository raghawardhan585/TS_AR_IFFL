#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 15 8 5 4 20 > System_15/MyRunInfo/Run_8.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
