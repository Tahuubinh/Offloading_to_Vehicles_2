import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
a=[]
b=[]
e=[]
fig, ax = plt.subplots(1)
for i in range(1,6):
    files=pd.read_csv("../../result4/Combine fuzzy and deep q "+str(i+1)+"/ketqua_oneday.csv")
    x=files["mean_reward"].to_numpy()[0:100]
    x = x[1:100]
    #print(x.shape)
    a.append(x)
    files=pd.read_csv("../../result4/deep q learning "+str(i)+"/ketqua_oneday.csv")
    xx=files["mean_reward"].to_numpy()[0:100]
    xx = xx[1:100]
    #print(xx.shape)
    b.append(xx)
    files=pd.read_csv("../DDQL_5phut"+str(i)+".csv")
    xxx=files["mean_reward"].to_numpy()[0:99]
    e.append(xxx)
    print(xxx.shape)
    #print(a)
m=[i for i in range(1,100)]


c=pd.read_csv("../../result4/fuzzy/fuzzy_150.csv")
d=pd.read_csv("../../result4/random/fuzzy_150.csv")

ax.plot(m,np.average(a,axis=0),marker='^', markevery=10,label='FDQO',color="orange")
ax.plot(m,np.average(b,axis=0),marker='o', markevery=10,label='DQL',color="blue")
ax.plot(m,np.average(e,axis=0),marker='x', markevery=10,label='DDQL',color="green")
# ax.fill_between(m, np.max(b,axis=0),np.min(b,axis=0), facecolor='#b9deff', alpha=0.5)

# # ax.plot(m,c["Fuzzy Controller"][0:100],marker='P', markevery=10,label='Fuzzy')
# # ax.plot(m,d["Random"][0:100],marker='x', markevery=10,label='Random')
# ax.fill_between(m, np.max(a,axis=0),np.min(a,axis=0), facecolor='#FFA107', alpha=0.5)
# ax.fill_between(m, np.max(c,axis=0),np.min(c,axis=0), facecolor='#b9deff', alpha=0.5)
ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=2)

plt.ylim(0,1)
#ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_xlabel('Time slots',fontsize=15)
ax.set_ylabel('Average QoE',fontsize=15)
plt.grid(alpha=0.5)
#plt.show()

plt.savefig("variation.pdf")