import pandas as pd
import numpy as np
import os
from config import *
class Fuzzy_Controller():

    def __init__(self):
      self.rule = pd.read_excel(os.path.join(DATA_DIR,"rule2.xlsx"),index_col = 0).to_numpy()
    
    def membership_function(self, a, b, c, d, value):
        if value <= a:
            return [1,0,0]
        elif value < b and value > a:
            return [(b-value) / (b-a), (value-a) / (b-a), 0]
        elif value >= b and value <= c:
            return [0,1,0]
        elif value > c and value < d:
            return [0, (d-value) / (d-c), (value-c) / (d-c)]
        else:
            return [0,0,1]

    def choose_action(self, observation):
        percent_action = [0,0,0,0,0]

        a = []
        server = self.membership_function(0.6, 1, 2.5, 4, observation[9])
        bus1_time = self.membership_function(0.5, 1.2, 2, 2.5, observation[1])  
        bus2_time = self.membership_function(0.5, 1.2, 2, 2.5, observation[4])
        bus3_time = self.membership_function(0.5, 1.2, 2, 2.5, observation[7])
        deadline = self.membership_function(1.2, 1.4, 2, 2.5, observation[13])
        a = [server, bus1_time, bus2_time,bus3_time, deadline]

        for i in range(0, len(self.rule)):
            xacsuat = 1
            for j in range(0, len(self.rule[i])-1):
                if self.rule[i][j] != -1 and self.rule[i][j] != -2:
                   xacsuat = xacsuat*a[j][self.rule[i][j]]
                elif self.rule[i][j] == -2:
                    xacsuat = xacsuat * (a[j][1]+a[j][2])
            percent_action[int(self.rule[i][-1])] += xacsuat

        return np.argmax(percent_action)