import sys
import os
import psutil
import subprocess
import time

total_memory = psutil.virtual_memory().total

print("Memory used before: %s" % psutil.virtual_memory().percent)

mem_usage_percent = []

process = subprocess.Popen(['python', 'basic_wrf_extraction.py', '--ymd', '20220101', '--hour', '01'])

count = 0
while psutil.pid_exists(process.pid):
    available_memory = psutil.virtual_memory().percent
    mem_usage_percent.append(available_memory)
    time.sleep(.01)
    count += 1
    if count > 1000: 
        break

process.wait()

max_percent_usage = max(mem_usage_percent)
print("Max mem used during process %s" % max_percent_usage)

# def main() :
    # probe to see what percentage a single file takes up
    # 


    