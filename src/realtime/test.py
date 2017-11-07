import os,time
current_milli_time1 = lambda: int(time.time() * 1000)
print(current_milli_time1())
t1 = current_milli_time1()
time.sleep(2.3)
current_milli_time2 = lambda: int(time.time() * 1000)
print(current_milli_time2())
t2 = current_milli_time2()
diff = t2-t1
print(diff/1000)