#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 21 0 4 4 8 > System_21/MyRunInfo/Run_0.txt & 
wait
python3 deepDMD.py '/cpu:0' 21 1 8 4 16 > System_21/MyRunInfo/Run_1.txt & 
wait
python3 deepDMD.py '/cpu:0' 21 2 12 4 24 > System_21/MyRunInfo/Run_2.txt & 
wait
python3 deepDMD.py '/cpu:0' 21 3 4 4 8 > System_21/MyRunInfo/Run_3.txt & 
wait
python3 deepDMD.py '/cpu:0' 21 4 8 4 16 > System_21/MyRunInfo/Run_4.txt & 
wait
python3 deepDMD.py '/cpu:0' 21 5 12 4 24 > System_21/MyRunInfo/Run_5.txt & 
wait
python3 deepDMD.py '/cpu:0' 22 2 4 4 8 > System_22/MyRunInfo/Run_2.txt & 
wait
python3 deepDMD.py '/cpu:0' 22 3 8 4 16 > System_22/MyRunInfo/Run_3.txt & 
wait
python3 deepDMD.py '/cpu:0' 22 4 12 4 24 > System_22/MyRunInfo/Run_4.txt & 
wait
python3 deepDMD.py '/cpu:0' 22 5 4 4 8 > System_22/MyRunInfo/Run_5.txt & 
wait
python3 deepDMD.py '/cpu:0' 22 6 8 4 16 > System_22/MyRunInfo/Run_6.txt & 
wait
python3 deepDMD.py '/cpu:0' 22 7 12 4 24 > System_22/MyRunInfo/Run_7.txt & 
wait
python3 deepDMD.py '/cpu:0' 23 2 4 4 8 > System_23/MyRunInfo/Run_2.txt & 
wait
python3 deepDMD.py '/cpu:0' 23 3 8 4 16 > System_23/MyRunInfo/Run_3.txt & 
wait
python3 deepDMD.py '/cpu:0' 23 4 12 4 24 > System_23/MyRunInfo/Run_4.txt & 
wait
python3 deepDMD.py '/cpu:0' 23 5 4 4 8 > System_23/MyRunInfo/Run_5.txt & 
wait
python3 deepDMD.py '/cpu:0' 23 6 8 4 16 > System_23/MyRunInfo/Run_6.txt & 
wait
python3 deepDMD.py '/cpu:0' 23 7 12 4 24 > System_23/MyRunInfo/Run_7.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
