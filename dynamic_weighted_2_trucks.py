import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from four_plus_truck_function import dyn_multi_opt
# from show_routes import CreateMap
from show_routes import CreateJSON



# Constants
B_TO_B = 100
B_TO_T = 10
N_WARDS = 3
N_TRUCKS = 2
W1 = 0.9
W2 = 0.1

# Set Random Seed
np.random.seed(42)

# Import Data
data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
for i in range(distance.shape[0]):
    distance.iloc[:, i] = distance.iloc[:, i]/np.max(distance.iloc[:, i])

# Optimization

# Ward 1 Optimization

data1 = data[data.Ward == 0]
visit1, visit2 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
visitedNodes = set()
obj_value1 = dyn_multi_opt(data1, [visit1, visit2], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 1', t_name = 'truck1', folder_Path = 'Data/Dynamic Data/Multiple Trucks/2 Trucks/', w1 = W1, w2 = W2, n_done = [0] * N_TRUCKS, n_trucks = N_TRUCKS)
print('\n Ward 1 Done \n')

# Ward 2 Optimization

data2 = data[data.Ward == 1]
visit1, visit2 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
visitedNodes = set()
obj_value2 = dyn_multi_opt(data2, [visit1, visit2], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 2', t_name = 'truck2', folder_Path = 'Data/Dynamic Data/Multiple Trucks/2 Trucks/', w1 = W1, w2 = W2, n_done = [0] * N_TRUCKS, n_trucks = N_TRUCKS)
print('\n Ward 2 Done \n')

# Ward 3 Optimization

data3 = data[data.Ward == 2]
visit1, visit2= (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
visitedNodes = set()
obj_value3 = dyn_multi_opt(data3, [visit1, visit2], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 3', t_name = 'truck3', folder_Path = 'Data/Dynamic Data/Multiple Trucks/2 Trucks/', w1 = W1, w2 = W2, n_done = [0] * N_TRUCKS, n_trucks = N_TRUCKS)
print('\n Ward 3 Done \n')


# Collect Data
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
path11 = []
path12 = []
path21 = []
path22 = []
path31 = []
path32 = []
v11 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 1/visited_truck1_1_{W1}_{W2}.csv')
v12 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 1/visited_truck1_2_{W1}_{W2}.csv')
v21 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 2/visited_truck2_1_{W1}_{W2}.csv')
v22 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 2/visited_truck2_2_{W1}_{W2}.csv')
v31 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 3/visited_truck3_1_{W1}_{W2}.csv')
v32 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 3/visited_truck3_2_{W1}_{W2}.csv')
v11.Node = v11.Node.astype('int')
v12.Node = v12.Node.astype('int')

v21.Node = v21.Node.astype('int')
v22.Node = v22.Node.astype('int')

v31.Node = v31.Node.astype('int')
v32.Node = v32.Node.astype('int')

for i in range(len(v11) - 1):
    path11.append((v11.iloc[i, 0], v11.iloc[i + 1, 0]))
for i in range(len(v12) - 1):
    path12.append((v12.iloc[i, 0], v12.iloc[i + 1, 0]))

for i in range(len(v21) - 1):
    path21.append((v21.iloc[i, 0], v21.iloc[i + 1, 0]))
for i in range(len(v22) - 1):
    path22.append((v22.iloc[i, 0], v22.iloc[i + 1, 0]))
    
for i in range(len(v31) - 1):
    path31.append((v31.iloc[i, 0], v31.iloc[i + 1, 0]))
for i in range(len(v32) - 1):
    path32.append((v32.iloc[i, 0], v32.iloc[i + 1, 0]))

gar11 = v11.iloc[-1,1]*10
gar12 = v12.iloc[-1,1]*10

gar21 = v21.iloc[-1,1]*10
gar22 = v22.iloc[-1,1]*10

gar31 = v31.iloc[-1,1]*10
gar32 = v32.iloc[-1,1]*10

dist11 = sum([distance.iloc[i,j] for i,j in path11])
dist12 = sum([distance.iloc[i,j] for i,j in path12])

dist21 = sum([distance.iloc[i,j] for i,j in path21])
dist22 = sum([distance.iloc[i,j] for i,j in path22])

dist31 = sum([distance.iloc[i,j] for i,j in path31])
dist32 = sum([distance.iloc[i,j] for i,j in path32])

v1 = [v11, v12]
v2 = [v21, v22]
v3 = [v31, v32]

print('--------------- SAVING STATISTICS ----------------------\n')
# Save Statistics

stats = pd.DataFrame(
    {
        'Fill Ward 1 (in %)' : [
            round(gar11, 4), 
            round(gar12, 4),
            '-'],
        'Garbage Fill Ward 1 (in Litres)' : [
            round(gar11/10 * B_TO_B, 4),
            round(gar12/10 * B_TO_B, 4),
            '-'],
        'Distance Travelled Ward 1 (in m)' : [
            round(dist11, 4),
            round(dist12, 4),
            '-'],
        'Garbage per Meter Ward 1 (in KG/m)' : [
            round(gar11/dist11, 4),
            round(gar12/dist12, 4),
            '-'],
        'Percentage of Bins covered Ward 1 (in %)' : [
            round( 100 * (v11.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (v12.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * np.sum([i.shape[0] - 2 for i in v1])/ data[data.Ward == 0].shape[0], 4)],
        'Fill Ward 2 (in %)' : [
            round(gar21, 4), 
            round(gar22, 4),
            '-'],
        'Garbage Fill Ward 2 (in Litres)' : [
            round(gar21/10 * B_TO_B, 4),
            round(gar22/10 * B_TO_B, 4),
            '-'],
        'Distance Travelled Ward 2 (in m)' : [
            round(dist21, 4),
            round(dist22, 4),
            '-'],
        'Garbage per Meter Ward 2 (in KG/m)' : [
            round(gar21/dist21, 4),
            round(gar22/dist22, 4),
            '-'],
        'Percentage of Bins covered Ward 2 (in %)' : [
            round( 100 * (v21.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * (v22.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * np.sum([i.shape[0] - 2 for i in v2])/ data[data.Ward == 1].shape[0], 4)],
        'Fill Ward 3 (in %)' : [
            round(gar31, 4), 
            round(gar32, 4),
            '-'],
        'Garbage Fill Ward 3 (in Litres)' : [
            round(gar31/10 * B_TO_B, 4),
            round(gar32/10 * B_TO_B, 4),
            '-'],
        'Distance Travelled Ward 3 (in m)' : [
            round(dist31, 4),
            round(dist32, 4),
            '-'],
        'Garbage per Meter Ward 3 (in KG/m)' : [
            round(gar31/dist31, 4),
            round(gar32/dist32, 4),
            '-'],
        'Percentage of Bins covered Ward 3 (in %)' : [
            round( 100 * (v31.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
            round( 100 * (v32.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
            round( 100 * np.sum([i.shape[0] - 2 for i in v3])/ data[data.Ward == 2].shape[0], 4)],
    }, index=['Truck 1', 'Truck 2', 'Total Percentage'])
stats.to_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Statistics.csv')

print('--------------- GENERATING MAP ----------------------')
# Plotting routes

# map = CreateMap()
# map.createRoutes('Data/Dynamic Data/Multiple Trucks/2 Trucks/', N_WARDS, N_TRUCKS, W1, W2, Multiple_truck = True)
# map.createLatLong('Data/Bin Locations.csv', N_WARDS)
# map.createRoutesDict(N_WARDS)
# map.addRoutesToMap(N_WARDS, N_TRUCKS)
# map.addDepot()
# map.addNodes('Data/Bin Locations.csv')
# map.saveMap('Data/Dynamic Data/Multiple Trucks/2 Trucks/')
# map.displayMap('Data/Dynamic Data/Multiple Trucks/2 Trucks/')

JSON = CreateJSON(N_WARDS, N_TRUCKS, W1, W2)
JSON.run('Data/Bin Locations.csv', 'Data/Dynamic Data/Multiple Trucks/2 Trucks/', True)