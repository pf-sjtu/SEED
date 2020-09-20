#coding:utf8
from memory_profiler import profile

@profile
def test1():
    c=0
    for item in range(100000):
        c+=1
    print(c)

if __name__=='__main__':
    test1()