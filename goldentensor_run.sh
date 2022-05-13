#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 31 0 8 4 16 > System_31/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 31 1 12 4 24 > System_31/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 32 0 8 4 16 > System_32/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 32 1 12 4 24 > System_32/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 33 0 8 4 16 > System_33/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 33 1 12 4 24 > System_33/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 34 0 8 4 16 > System_34/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 34 1 12 4 24 > System_34/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 35 0 8 4 16 > System_35/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 35 1 12 4 24 > System_35/MyRunInfo/Run_1.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
