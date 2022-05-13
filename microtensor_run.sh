#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 41 0 8 4 16 > System_41/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 41 1 12 4 24 > System_41/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 42 0 8 4 16 > System_42/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 42 1 12 4 24 > System_42/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 43 0 8 4 16 > System_43/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 43 1 12 4 24 > System_43/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 44 0 8 4 16 > System_44/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 44 1 12 4 24 > System_44/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 45 0 8 4 16 > System_45/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 45 1 12 4 24 > System_45/MyRunInfo/Run_1.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
