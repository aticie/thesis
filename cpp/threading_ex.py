
import threading 
import time 
import sys 
  
def func(): 
    while True: 
        time.sleep(0.5) 
        print('Thread alive, and it won\'t die on program termination') 
  
t1 = threading.Thread(target=func) 
t1.start() 
time.sleep(2) 
sys.exit() 
