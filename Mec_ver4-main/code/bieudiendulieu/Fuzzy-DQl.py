import pandas as pd
import matplotlib.pyplot as plt
number_tasks_per_episode = pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/n_quality_tasks_dql.csv").to_numpy()
kq = pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/kq.csv").to_numpy()
x = 0
m , n = [], []
for i in range(100):
    fuzzy, dql = 0, 0
    n_tasks = sum(number_tasks_per_episode[i]) 
    for i in range(n_tasks):
        if kq[x+i] == 1:
           fuzzy +=1
        else:
           dql += 1 
    m.append(fuzzy/n_tasks)
    n.append(dql/n_tasks)
    x = x+n_tasks
aa =[]
for i in range(100):
    aa.append([m[i],n[i]])
print(aa)
df = pd.DataFrame(aa,columns = ["Fuzzy","DQL"])
df.plot.line()
plt.savefig("F-Qratio.eps")

plt.show()

print(df)