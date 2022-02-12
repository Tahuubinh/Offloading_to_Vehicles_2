import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
files=pd.read_excel("C:/Users/vutri/OneDrive/Desktop/15092020/code/data/data900.xlsx")
x=[i for i in range(1,130)]
fig, ax = plt.subplots()
ax.plot(files["x"],files["y"])
circle1=plt.Circle((40500,165000),2624,color="blue")
circle2=plt.Circle((42500,160000),2624,color="blue")
circle3=plt.Circle((45000,153000),2624,color="blue")
circle4=plt.Circle((48000,155000),2624,color="blue")
circle5=plt.Circle((54000,153500),2624,color="blue")
circle6=plt.Circle((46250,162000),2624,color="blue")
circle7=plt.Circle((52500,148000),2624,color="blue")
circle8=plt.Circle((51000,160300),2624,color="blue")
circle9=plt.Circle((50000,173000),2624,color="blue")
circle10=plt.Circle((47000,169000),2624,color="blue")
circle11=plt.Circle((43500,157000),2624,color="blue")
circle12=plt.Circle((51000,155000),2624,color="blue")
circle14=plt.Circle((49000,165000),2624,color="blue")
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

ax.add_artist(circle4)
ax.add_artist(circle5)
ax.add_artist(circle6)

ax.add_artist(circle7)
ax.add_artist(circle8)
ax.add_artist(circle9)

ax.add_artist(circle10)
ax.add_artist(circle11)
ax.add_artist(circle12)
ax.add_artist(circle14)
#plt.show()
xy=[[40500,165000],[42500,160000],[45000,153000],[48000,155000],[54000,153500],\
    [46250,162000],[52500,148000],[51000,160300],[50000,173000],[47000,169000],\
    [43500,157000],[51000,155000],[49000,165000]]
aaa=[]
print(files["x"][2])
for j in range(len(files["x"])):
    minn=10
    for i in range(13):
        if minn>(np.sqrt(np.power(xy[i][0]-files["x"][j],2)+np.power(xy[i][1]-files["y"][j],2)))*0.0003048:
            minn=(np.sqrt(np.power(xy[i][0]-files["x"][j],2)+np.power(xy[i][1]-files["y"][j],2)))*0.0003048
    aaa.append(minn)
files["minmin"]=aaa
files.to_excel("data9000.xlsx")
