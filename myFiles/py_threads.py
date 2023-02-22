import logging
import threading
import time
from datetime import date, timedelta, datetime

def thread_function(name):
    logging.info("thread %s starting", name)
    time.sleep(2)
    logging.info("thread %s finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("Main       : before creating thread")
    begin_date = "20230131"  # format as "yyyymmdd"
    end_date = "20230202"
    begin_hour = "00:00"
    end_hour = "23:00"
    begin_day_dt = datetime.strptime(begin_date, "%Y%m%d")
    end_day_dt = datetime.strptime(end_date, "%Y%m%d")
    begin_hour_dt = datetime.strptime(begin_hour, "%H:%M")
    end_hour_dt = datetime.strptime(end_hour, "%H:%M")
    hour_range = (end_hour_dt - begin_hour_dt).seconds // (60 * 60)
    date_range = (end_day_dt - begin_day_dt).days
    threads = []
    for i in range(0, date_range):
        for j in range(0, hour_range):
            dtobj = begin_day_dt + timedelta(days=i, hours=j)
            t = threading.Thread(target=thread_function, args=(dtobj,))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()


    logging.info("Main       : all donw")