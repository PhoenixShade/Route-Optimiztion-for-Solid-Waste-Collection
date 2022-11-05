import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from dynamic_function import dyn_opt
from show_routes import CreateMap

# Constants
B_TO_B = 100
B_TO_T = 10
N_WARDS = 3
N_TRUCKS = 1

# Set Random Seed
np.random.seed(42)

# Import Data
data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
for i in range(distance.shape[0]):
    distance.iloc[:, i] = distance.iloc[:, i]/np.max(distance.iloc[:, i])

# Add Fill_ratio, distance and fill per meter
fill_ratio = [0.0] + [np.random.rand() for i in range(data.shape[0] - 1)]
distance_from_0 = distance.iloc[:, 0]
data['fill_ratio'] = fill_ratio
data['distance_from_0'] = distance_from_0
fill_p_m = [0.0] + list(B_TO_B * data.loc[1:, 'fill_ratio'] / data.loc[1:, 'distance_from_0'])
data['fill_p_m'] = fill_p_m

# Optimization
obj_values = []
for i in range(11):
    w1, w2 = round(i/10, 1), round(1 - i/10, 1)
    print(f"\n----------------- Processing w1 : {w1} | w2 : {w2} -----------------")
    visit1, visit2, visit3 = (
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        )
    data1 = data[data.Ward == 0]
    data2 = data[data.Ward == 1]
    data3 = data[data.Ward == 2]

    obj_value = dyn_opt(data1, data2, data3, distance, folder_path = 'Data/Dynamic Data/Weight Finding/', w1 = w1, w2 = w2, visit1 = visit1, visit2 = visit2, visit3 = visit3)
    obj_values.append(np.sum(obj_value))

# Plotting

w1s = [round(i/10, 1) for i in range(0, 11)]
Figure = figure(figsize=(15, 15))
plt.scatter(w1s, obj_values, c = 'blue')
plt.scatter([round(1 - i, 1) for i in w1s], obj_values, c = 'red')
plt.title('Objective VS W1')
plt.xlabel('Value of Weights')
plt.ylabel('Objective Value')
plt.legend(['W1', 'W2'])
plt.show()



W1 = w1s[np.argmin(obj_values)]
W2 = round(1 - W1, 1)
print(f"Best w1 value is : {W1}.")
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
# Storing statistics of best case
v11 = pd.read_csv(f'Data/Dynamic Data/Weight Finding/Visited Truck 1/visited_truck1_{W1}_{W2}.csv')
v21 = pd.read_csv(f'Data/Dynamic Data/Weight Finding/Visited Truck 2/visited_truck2_{W1}_{W2}.csv')
v31 = pd.read_csv(f'Data/Dynamic Data/Weight Finding/Visited Truck 3/visited_truck3_{W1}_{W2}.csv')
v11.Node = v11.Node.astype('int')
v21.Node = v21.Node.astype('int')
v31.Node = v31.Node.astype('int')
path11 = []
path21 = []
path31 = []
for i in range(len(v11) - 1):
    path11.append((v11.iloc[i, 0], v11.iloc[i + 1, 0]))
for i in range(len(v21) - 1):
    path21.append((v21.iloc[i, 0], v21.iloc[i + 1, 0]))
for i in range(len(v31) - 1):
    path31.append((v31.iloc[i, 0], v31.iloc[i + 1, 0]))
gar11 = v11.iloc[-1,1]*10
gar21 = v21.iloc[-1,1]*10
gar31 = v31.iloc[-1,1]*10
dist11 = sum([distance.iloc[i,j] for i,j in path11])
dist21 = sum([distance.iloc[i,j] for i,j in path21])
dist31 = sum([distance.iloc[i,j] for i,j in path31])


v1 = [v11]
v2 = [v21]
v3 = [v31]
# Save Statistics

print('--------------- SAVING STATISTICS ----------------------\n')

stats = pd.DataFrame(
    {
        'Fill Ward 1 (in %)' : [
            round(gar11, 4), 
            '-'],
        'Garbage Fill Ward 1 (in Litres)' : [
            round(gar11/10 * B_TO_B, 4),
            '-'],
        'Distance Travelled Ward 1 (in m)' : [
            round(dist11, 4),
            '-'],
        'Garbage per Meter Ward 1 (in KG/m)' : [
            round(gar11/dist11, 4),
            '-'],
        'Percentage of Bins covered Ward 1 (in %)' : [
            round( 100 * (v11.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * np.sum([i.shape[0] - 2 for i in v1])/ data[data.Ward == 0].shape[0], 4)],
        'Fill Ward 2 (in %)' : [
            round(gar21, 4), 
            '-'],
        'Garbage Fill Ward 2 (in Litres)' : [
            round(gar21/10 * B_TO_B, 4),
            '-'],
        'Distance Travelled Ward 2 (in m)' : [
            round(dist21, 4),
            '-'],
        'Garbage per Meter Ward 2 (in KG/m)' : [
            round(gar21/dist21, 4),
            '-'],
        'Percentage of Bins covered Ward 2 (in %)' : [
            round( 100 * (v21.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * np.sum([i.shape[0] - 2 for i in v2])/ data[data.Ward == 1].shape[0], 4)],
        'Fill Ward 3 (in %)' : [
            round(gar31, 4), 
            '-'],
        'Garbage Fill Ward 3 (in Litres)' : [
            round(gar31/10 * B_TO_B, 4),
            '-'],
        'Distance Travelled Ward 3 (in m)' : [
            round(dist31, 4),
            '-'],
        'Garbage per Meter Ward 3 (in KG/m)' : [
            round(gar31/dist31, 4),
            '-'],
        'Percentage of Bins covered Ward 3 (in %)' : [
            round( 100 * (v31.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
            round( 100 * np.sum([i.shape[0] - 2 for i in v3])/ data[data.Ward == 2].shape[0], 4)],
    }, index=['Truck 1', 'Total Percentage'])
stats.to_csv('Data/Dynamic Data/Weight Finding/Statistics.csv')

print('--------------- GENERATING MAP ----------------------')

# Plotting routes

map = CreateMap()
map.createRoutes('Data/Dynamic Data/Weight Finding/', N_WARDS, N_TRUCKS, W1, W2)
map.createLatLong('Data/Bin Locations.csv', N_WARDS)
map.createRoutesDict(N_WARDS)
map.addRoutesToMap(N_WARDS, N_TRUCKS)
map.addDepot()
map.addNodes('Data/Bin Locations.csv')
map.saveMap('Data/Dynamic Data/Weight Finding/')
map.displayMap('Data/Dynamic Data/Weight Finding/')