import os
import numpy as np
import pandas as pd

os.makedirs(os.path.join('.','data'), exist_ok=True)
data_file = os.path.join('.','data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 用平均值来替代空值
inputs_rooms,inputs_alley, outputs = data.iloc[:,0],data.iloc[:,1], data.iloc[:,-1]
inputs_rooms = inputs_rooms.fillna(inputs_rooms.mean())
print(inputs_rooms)

inputs_alley = pd.get_dummies(inputs_alley,dummy_na=True)
print(inputs_alley)

import torch

inputs = torch.tensor(pd.concat([inputs_rooms,inputs_alley], axis=1).to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(inputs,y)

