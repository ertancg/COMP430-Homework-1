#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:57:56 2022

@author: ecguner
"""
import sys
import random
import subprocess as sp
import time
import multiprocessing

k_indexes = [4, 8, 16, 32, 64, 128]
anonymization = ["clustering"]
#clustering DGHs adult-test-2.csv clustering-test-2.csv 4
command_base = "python3 skeleton.py "
#os.system("python3 skeleton.py random DGHs adult-test-2.csv testtest.csv 4 10")
def procs(command_base, arg, idx):
    try:
        retcode = sp.call(command_base + arg, shell=True)
        time.sleep(2)
        print("Run #", idx)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)
def run_all_proc():
    for algo in anonymization:
        for k in k_indexes:
            print("Algorithm: " + algo + ", k = " + str(k))
            process_list = []
            for i in range(1, 4):
                if algo == "random":
                    command = algo + " DGHs adult-hw1.csv " + algo + "_" + "k=" + str(k) + "_" + str(i) + ".csv" + " " + str(k) + " " + str(random.randint(1, 100)) 
                else:
                    command = algo + " DGHs adult-hw1.csv " + algo + "_" + "k=" + str(k) + "_" + str(i) + ".csv" + " " + str(k)

                p = multiprocessing.Process(target=procs, args=(command_base, command, i))
                process_list.append(p)
                p.start()
                
            for p in process_list:
                p.join()
                
                print('-' * 25)
            print('#' * 50)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_all_proc()


   