import numpy as np
import pandas as pd

# CONSTANTS
B_TO_B = 100
B_TO_T = 10
N_WARDS = 3

def updateFill(df, m):
    fill_ratio_prev = 'fill_ratio_' + str(m - 1)
    fill_ratio_new = 'fill_ratio_' + str(m)
    flag = False
    if m == 0:
        fill_ratio = [np.random.rand() for _ in range(len(df))]
    else:
        fill_ratio = df.loc[:, fill_ratio_prev].tolist()
        for i in range(len(fill_ratio)):
            if np.random.rand() < 0.8:
                fill_ratio[i] = fill_ratio[i] + np.random.uniform(0, 1 - fill_ratio[i]) / 10
    df.insert(df.shape[1], fill_ratio_new, fill_ratio)
    m += 1
    if [1.0]*df.shape[0] == fill_ratio : flag = True
    if flag : return df
    else : updateFill(df, m)

data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
for ward in range(N_WARDS):
    np.random.seed(42)
    DF = data[data.Ward == ward]
    m = 0
    df = updateFill(DF, m)
    df.to_csv(f'Images/Bins Filling Ward {ward + 1}.csv', index = False)


