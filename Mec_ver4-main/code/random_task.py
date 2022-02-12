import random as rd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import exp, factorial
path =os.path.abspath(__file__)
path =Path(path).parent.parent

# a = np.zeros(200)
# xinc = 25
# yinc = 1400
# for i in range(200):
#     a[i] = (10 - abs(i % 40 - 20)) * xinc + yinc

# a = np.ones(200)
# lambda_value = 12
# for i in range(200):
#     j = i % 10 + 6
#     a[i] = exp(-lambda_value) * (lambda_value**j) / factorial(j)*16000

files = open("{}/{}/summary.csv".format(str(path),"data_task"),"w")
files.write("number of tasks\n")

for i in range(200):
    with open("{}/{}/datatask{}.csv".format(str(path),"data_task",i),"w") as output:
        # indexs=rd.randint(900,1200)
        # indexs=rd.randint(int(a[i]-50),int(a[i]+50))
        indexs=rd.randint(1200,1400)
        # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
        randomNums = np.random.normal(i*300+150, 50, size=indexs)
        m = np.round(randomNums)
        m = np.sort(m)
        for j in range(len(m)):
            if m[j] < i*300:
                m[j] = i*300
            if m[j] > (i+1)*300 - 1:
                m[j] = (i+1)*300 - 1
        m1 = np.random.randint(1000,2000,indexs)
        m2 = np.random.randint(100,200,indexs)
        m3 = np.random.randint(500,1500,indexs)
        m4 = 1+np.random.rand(indexs)*2
        for j in range(indexs):
            output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))
    files.write("{}\n".format(indexs))
    #import pdb;pdb.set_trace()
files.close()

number_tasks = pd.read_csv("../data_task/summary.csv").values
b = np.arange(200)
plt.plot(b, number_tasks)
plt.axis([0, 100, 0, 2100])
plt.show()

















