#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 15 45 8 4 16 > System_15/MyRunInfo/Run_45.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 46 8 5 16 > System_15/MyRunInfo/Run_46.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 47 10 4 20 > System_15/MyRunInfo/Run_47.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 48 10 5 20 > System_15/MyRunInfo/Run_48.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 49 12 4 24 > System_15/MyRunInfo/Run_49.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 50 12 5 24 > System_15/MyRunInfo/Run_50.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 51 15 4 30 > System_15/MyRunInfo/Run_51.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 52 15 5 30 > System_15/MyRunInfo/Run_52.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 53 18 4 36 > System_15/MyRunInfo/Run_53.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 54 18 5 36 > System_15/MyRunInfo/Run_54.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 55 21 4 42 > System_15/MyRunInfo/Run_55.txt & 
wait
python3 deepDMD.py '/cpu:0' 15 56 21 5 42 > System_15/MyRunInfo/Run_56.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
