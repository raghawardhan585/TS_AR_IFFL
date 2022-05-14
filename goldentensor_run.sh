#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 15 21 4 3 8 > System_15/MyRunInfo/Run_21.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 22 4 4 8 > System_15/MyRunInfo/Run_22.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 23 4 5 8 > System_15/MyRunInfo/Run_23.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 24 6 3 12 > System_15/MyRunInfo/Run_24.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 25 6 4 12 > System_15/MyRunInfo/Run_25.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 26 6 5 12 > System_15/MyRunInfo/Run_26.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 27 8 3 16 > System_15/MyRunInfo/Run_27.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 28 8 4 16 > System_15/MyRunInfo/Run_28.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 29 8 5 16 > System_15/MyRunInfo/Run_29.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 30 10 3 20 > System_15/MyRunInfo/Run_30.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 31 10 4 20 > System_15/MyRunInfo/Run_31.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 32 10 5 20 > System_15/MyRunInfo/Run_32.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
