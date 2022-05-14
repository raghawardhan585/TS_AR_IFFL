#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 15 9 4 3 8 > System_15/MyRunInfo/Run_9.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 10 4 4 8 > System_15/MyRunInfo/Run_10.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 11 4 5 8 > System_15/MyRunInfo/Run_11.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 12 6 3 12 > System_15/MyRunInfo/Run_12.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 13 6 4 12 > System_15/MyRunInfo/Run_13.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 14 6 5 12 > System_15/MyRunInfo/Run_14.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 15 8 3 16 > System_15/MyRunInfo/Run_15.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 16 8 4 16 > System_15/MyRunInfo/Run_16.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 17 8 5 16 > System_15/MyRunInfo/Run_17.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 18 10 3 20 > System_15/MyRunInfo/Run_18.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 19 10 4 20 > System_15/MyRunInfo/Run_19.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 20 10 5 20 > System_15/MyRunInfo/Run_20.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
