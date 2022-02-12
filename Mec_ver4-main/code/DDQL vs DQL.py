import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ddql = pd.read_csv("DDQL_5phut.csv").values
dql = pd.read_csv("FDQO_5phut.csv").values

totalddql = ddql[:,0].reshape([-1,1])
meanddql = ddql[:,1].reshape([-1,1])

totaldql = dql[0:99,0].reshape([-1,1])
meandql = dql[0:99,1].reshape([-1,1])
x = [i + 1 for i in range(99)]
x = np.array(x).reshape([-1,1])

#print (meanddql) 
plt.plot(x, totalddql)
plt.plot(x, totaldql)

plt.show()