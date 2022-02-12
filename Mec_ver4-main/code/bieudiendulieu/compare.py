import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

a=[]
b=[]
for i in range(1,2):
    files=pd.read_csv("FDQO_5phut.csv")
    x=files["mean_reward"].to_numpy()[0:100]
    a.append(x)
    files=pd.read_csv("DQL_5phut.csv")
    xx=files["mean_reward"].to_numpy()[0:100]
    b.append(xx)
    #print(a)
files=pd.read_csv("Fuzzy_5phut.csv")
files1=pd.read_csv("MAB_5phut.csv")
#files=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/deep q learning 1/ketqua_oneday.csv")
#files1=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/deep q learning 2/ketqua_oneday.csv")
m=[i for i in range(1,101)]
#files=pd.read_excel("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/compare.xlsx")
fig, ax = plt.subplots()
x=[i for i in range(1,101)]
labels=["an","an1","an2","an3"]

ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="FDQO",color="orange",lw=1)
ax.plot(x, files1["mean_reward"][0:100],marker="P", markevery=5,color="red",label="ε-greedy",lw=1)
ax.plot(x, files["mean_reward"][0:100],marker='o', markevery=5,label="Fuzzy",color="green",lw=1)
ax.plot(x, np.average(b,axis=0)[0:100],marker="o", markevery=5,color="blue",label="DQL",lw=1)
ax.set_ylabel('Average QoE',fontsize=16)
ax.set_xlabel('Time slots',fontsize=16)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])

plt.setp(ax.get_xticklabels(), fontsize=15)
plt.setp(ax.get_yticklabels(), fontsize=15)
#ax.set_xticklabels(labels)
plt.legend()
plt.grid(alpha=0.5)
#loc='upper center'
#plt.show()
plt.savefig("Compare_5p.eps")
#print(max(files1["Random"]))
#print(min(files1["Random"]))
#print(np.average(b,axis=0))"""
