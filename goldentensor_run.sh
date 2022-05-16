#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 15 33 8 4 16 > System_15/MyRunInfo/Run_33.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 34 8 5 16 > System_15/MyRunInfo/Run_34.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 35 10 4 20 > System_15/MyRunInfo/Run_35.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 36 10 5 20 > System_15/MyRunInfo/Run_36.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 37 12 4 24 > System_15/MyRunInfo/Run_37.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 38 12 5 24 > System_15/MyRunInfo/Run_38.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 39 15 4 30 > System_15/MyRunInfo/Run_39.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 40 15 5 30 > System_15/MyRunInfo/Run_40.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 41 18 4 36 > System_15/MyRunInfo/Run_41.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 42 18 5 36 > System_15/MyRunInfo/Run_42.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 43 21 4 42 > System_15/MyRunInfo/Run_43.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 44 21 5 42 > System_15/MyRunInfo/Run_44.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
