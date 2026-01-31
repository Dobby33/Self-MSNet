import time
import os
import random


class Log(object):
    def __init__(self, time_stamp, name='test_log', path="."):
        self.log_file_name = "%s/" % path + name + "_" + time_stamp + ".log"

 
    @classmethod
    def get_format_cur_time(self):
        return time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()));
    
    def get_cur_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()));
 
    def log_print(self, msg_content, msg_type="log"):
        cur_time = self.get_cur_time()
        line = "" + cur_time + "[" + msg_type + "] " + msg_content
        print (line)
        f = open(self.log_file_name, "a+")
        f.write(line + "\n" )
        f.close()
 
    def info(self, msg):
        self.log_print("info", msg)
 
    def warn(self, msg):
        self.log_print("warn", msg)
 
    def error(self, msg):
        self.log_print("error", msg)
