import numpy as np
import pandas as pd
import requests, json

def distance(i, j):
  r = requests.get(f"http://router.project-osrm.org/route/v1/car/{i[0]},{i[1]};{j[0]},{j[1]}?overview=false""")
  return json.loads(r.content).get('routes')[0]['distance']

df = pd.read_csv('Data/Bin Locations.csv', index_col='id').sort_index()
dist_mat = np.zeros((df.shape[0], df.shape[0]))
count = 0
for i in range(df.shape[0]):
    for j in range(df.shape[0]):
        print(f"-------Processing : {i} | {j}-------")
        if i == j:
            dist_mat[i, j] = 0.0
        elif i < j:
            if (df.iloc[i].Ward == df.iloc[j].Ward) or (df.iloc[i].Ward == -1) or (df.iloc[j].Ward == -1):
                st_point = (df.iloc[i].x, df.iloc[i].y)
                ed_point = (df.iloc[j].x, df.iloc[j].y)
                dist_mat[i, j] = distance(st_point, ed_point)
                count += 1
            else:
                dist_mat[i, j] == np.nan
        else:
            dist_mat[i, j] = dist_mat[j, i]
pd.DataFrame(dist_mat, index=df.index.tolist(), columns=df.index.tolist()).to_csv('Data/distance.csv')