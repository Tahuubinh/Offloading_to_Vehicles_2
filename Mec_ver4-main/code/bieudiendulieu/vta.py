import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""
files=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/a/data7/data_csv1.csv")
files=files.sort_values(by=['time'])

for i in range(128):
    files1=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/a/data7/data_csv"+str(i+2)+".csv")
    #data=np.sort(files1, axis = 0)
    files1=files1.sort_values(by=['time'])
    files=files.append(files1,ignore_index = True)
    #print(data)
files.to_excel("data_csv.xlsx")

"""
files=pd.read_excel("C:/Users/vutri/OneDrive/Desktop/15092020/code/code/bieudiendulieu/data_csv.xlsx")
file1=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result3/random/chiatask.csv")
print(files)
print(file1)
files["may"]=file1["somay"]
files["distance"]=file1["distance"]
files.to_excel("data_rd.xlsx")