#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 24 2 4 4 8 > System_24/MyRunInfo/Run_2.txt & 
wait
python3 deepDMD.py '/cpu:0' 24 3 8 4 16 > System_24/MyRunInfo/Run_3.txt & 
wait
python3 deepDMD.py '/cpu:0' 24 4 12 4 24 > System_24/MyRunInfo/Run_4.txt & 
wait
python3 deepDMD.py '/cpu:0' 24 5 12 4 24 > System_24/MyRunInfo/Run_5.txt & 
wait
python3 deepDMD.py '/cpu:0' 25 0 4 4 8 > System_25/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 25 1 8 4 16 > System_25/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 25 2 12 4 24 > System_25/MyRunInfo/Run_2.txt & 
wait
python3 deepDMD.py '/cpu:0' 25 3 12 4 24 > System_25/MyRunInfo/Run_3.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
