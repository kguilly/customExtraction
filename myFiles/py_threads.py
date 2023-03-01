import logging
import threading
import time
from datetime import date, timedelta, datetime
import queue

def thread_function(arr, i):
    time.sleep(2)
    x = 50*i
    y = x+15
    z = y / 55.0
    time.sleep(2)
    q.put(z)

if __name__ == "__main__":
    threads = []
    arr = ['first elements of the array']
    q = queue.Queue()
    for i in range(0, 15):
        t = threading.Thread(target=thread_function, args=(arr,i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    while not q.empty():
        arr.append(q.get())

    print(arr)
