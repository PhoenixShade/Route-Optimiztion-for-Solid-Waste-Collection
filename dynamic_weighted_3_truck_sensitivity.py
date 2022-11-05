import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from four_plus_truck_function import dyn_multi_opt
from show_routes import CreateMap


# Constants
B_TO_B = 100
B_TO_T = 10
N_WARDS = 3
N_TRUCKS = 3
w1s = [round(i/10, 1) for i in range(0, 11)]
total_garbage = []
total_distance = []

distance_per_ward_11 = []
garbage_per_ward_11 = []
distance_per_ward_12 = []
garbage_per_ward_12 = []
distance_per_ward_13 = []
garbage_per_ward_13 = []
distance_per_ward_21 = []
garbage_per_ward_21 = []
distance_per_ward_22 = []
garbage_per_ward_22 = []
distance_per_ward_23 = []
garbage_per_ward_23 = []
distance_per_ward_31 = []
garbage_per_ward_31 = []
distance_per_ward_32 = []
garbage_per_ward_32 = []
distance_per_ward_33 = []
garbage_per_ward_33 = []


# Set Random Seed
# np.random.seed(42)

# Import Data
data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
for i in range(distance.shape[0]):
    distance.iloc[:, i] = distance.iloc[:, i]/np.max(distance.iloc[:, i])


for w1 in w1s:
    np.random.seed(42)
    W1 = w1
    W2 = round(1 - W1, 1)
    # Optimization
    print(f"------------------- Processing {W1} and {W2} ----------------------")
    # Ward 1 Optimization

    data1 = data[data.Ward == 0]
    visit1, visit2, visit3 = (
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        )
    visitedNodes = set()
    obj_value1 = dyn_multi_opt(data1, [visit1, visit2, visit3], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 1', t_name = 'truck1', folder_Path = 'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/', w1 = W1, w2 = W2, n_done = [0] * N_TRUCKS, n_trucks = N_TRUCKS)
    print('\n Ward 1 Done \n')

    # Ward 2 Optimization

    data2 = data[data.Ward == 1]
    visit1, visit2, visit3 = (
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}),
        )
    visitedNodes = set()
    obj_value2 = dyn_multi_opt(data2, [visit1, visit2, visit3], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 2', t_name = 'truck2', folder_Path = 'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/', w1 = W1, w2 = W2, n_done = [0] * N_TRUCKS, n_trucks = N_TRUCKS)
    print('\n Ward 2 Done \n')

    # Ward 3 Optimization

    data3 = data[data.Ward == 2]
    visit1, visit2, visit3 = (
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        )
    visitedNodes = set()
    obj_value3 = dyn_multi_opt(data3, [visit1, visit2, visit3], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 3', t_name = 'truck3', folder_Path = 'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/', w1 = W1, w2 = W2, n_done = [0] * N_TRUCKS, n_trucks = N_TRUCKS)
    print('\n Ward 3 Done \n')


    # Collect Data
    distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
    path11 = []
    path12 = []
    path13 = []

    path21 = []
    path22 = []
    path23 = []

    path31 = []
    path32 = []
    path33 = []

    v11 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 1/visited_truck1_1_{W1}_{W2}.csv')
    v12 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 1/visited_truck1_2_{W1}_{W2}.csv')
    v13 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 1/visited_truck1_3_{W1}_{W2}.csv')

    v21 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 2/visited_truck2_1_{W1}_{W2}.csv')
    v22 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 2/visited_truck2_2_{W1}_{W2}.csv')
    v23 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 2/visited_truck2_3_{W1}_{W2}.csv')

    v31 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 3/visited_truck3_1_{W1}_{W2}.csv')
    v32 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 3/visited_truck3_2_{W1}_{W2}.csv')
    v33 = pd.read_csv(f'Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Visited Truck 3/visited_truck3_3_{W1}_{W2}.csv')

    v11.Node = v11.Node.astype('int')
    v12.Node = v12.Node.astype('int')
    v13.Node = v13.Node.astype('int')

    v21.Node = v21.Node.astype('int')
    v22.Node = v22.Node.astype('int')
    v23.Node = v23.Node.astype('int')

    v31.Node = v31.Node.astype('int')
    v32.Node = v32.Node.astype('int')
    v33.Node = v33.Node.astype('int')

    for i in range(len(v11) - 1):
        path11.append((v11.iloc[i, 0], v11.iloc[i + 1, 0]))
    for i in range(len(v12) - 1):
        path12.append((v12.iloc[i, 0], v12.iloc[i + 1, 0]))
    for i in range(len(v13) - 1):
        path13.append((v13.iloc[i, 0], v13.iloc[i + 1, 0]))

    for i in range(len(v21) - 1):
        path21.append((v21.iloc[i, 0], v21.iloc[i + 1, 0]))
    for i in range(len(v22) - 1):
        path22.append((v22.iloc[i, 0], v22.iloc[i + 1, 0]))
    for i in range(len(v23) - 1):
        path23.append((v23.iloc[i, 0], v23.iloc[i + 1, 0]))

    for i in range(len(v31) - 1):
        path31.append((v31.iloc[i, 0], v31.iloc[i + 1, 0]))
    for i in range(len(v32) - 1):
        path32.append((v32.iloc[i, 0], v32.iloc[i + 1, 0]))
    for i in range(len(v33) - 1):
        path33.append((v33.iloc[i, 0], v33.iloc[i + 1, 0]))

    gar11 = v11.iloc[-1,1]*100
    gar12 = v12.iloc[-1,1]*100
    gar13 = v13.iloc[-1,1]*100

    gar21 = v21.iloc[-1,1]*100
    gar22 = v22.iloc[-1,1]*100
    gar23 = v23.iloc[-1,1]*100

    gar31 = v31.iloc[-1,1]*100
    gar32 = v32.iloc[-1,1]*100
    gar33 = v33.iloc[-1,1]*100

    dist11 = sum([distance.iloc[i,j] for i,j in path11])/1000
    dist12 = sum([distance.iloc[i,j] for i,j in path12])/1000
    dist13 = sum([distance.iloc[i,j] for i,j in path13])/1000

    dist21 = sum([distance.iloc[i,j] for i,j in path21])/1000
    dist22 = sum([distance.iloc[i,j] for i,j in path22])/1000
    dist23 = sum([distance.iloc[i,j] for i,j in path23])/1000

    dist31 = sum([distance.iloc[i,j] for i,j in path31])/1000
    dist32 = sum([distance.iloc[i,j] for i,j in path32])/1000
    dist33 = sum([distance.iloc[i,j] for i,j in path33])/1000

    total_garbage.append(gar11 + gar12 + gar13 + gar21 + gar22 + gar23 + gar31 + gar32 + gar33)
    total_distance.append(dist11 + dist12 + dist13 + dist21 + dist22 + dist23 + dist31 + dist32 + dist33)
    
    garbage_per_ward_11.append(gar11)
    garbage_per_ward_12.append(gar12)
    garbage_per_ward_13.append(gar13)

    garbage_per_ward_21.append(gar21)
    garbage_per_ward_22.append(gar22)
    garbage_per_ward_23.append(gar23)
    
    garbage_per_ward_31.append(gar31)
    garbage_per_ward_32.append(gar32)
    garbage_per_ward_33.append(gar33)

    distance_per_ward_11.append(dist11)
    distance_per_ward_12.append(dist12)
    distance_per_ward_13.append(dist13)
    
    distance_per_ward_21.append(dist21)
    distance_per_ward_22.append(dist22)
    distance_per_ward_23.append(dist23)

    distance_per_ward_31.append(dist31)
    distance_per_ward_32.append(dist32)
    distance_per_ward_33.append(dist33)


temp = pd.DataFrame({'W1' : w1s, 'Garbage' : total_garbage, 'Distance' : total_distance})
temp.to_csv('Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Statistics.csv', index = False)

temp2 = pd.DataFrame({
    'W1' : w1s, 
    'Garbage Region 1 Truck 1' : garbage_per_ward_11,
    'Distance Region 1 Truck 1' : distance_per_ward_11,
    'Garbage Region 1 Truck 2' : garbage_per_ward_12,
    'Distance Region 1 Truck 2' : distance_per_ward_12,
    'Garbage Region 1 Truck 3' : garbage_per_ward_13,
    'Distance Region 1 Truck 3' : distance_per_ward_13,
    'Garbage Region 2 Truck 1' : garbage_per_ward_21,
    'Distance Region 2 Truck 1' : distance_per_ward_21,
    'Garbage Region 2 Truck 2' : garbage_per_ward_22,
    'Distance Region 2 Truck 2' : distance_per_ward_22,
    'Garbage Region 2 Truck 3' : garbage_per_ward_23,
    'Distance Region 2 Truck 3' : distance_per_ward_23,
    'Garbage Region 3 Truck 1' : garbage_per_ward_31,
    'Distance Region 3 Truck 1' : distance_per_ward_31,
    'Garbage Region 3 Truck 2' : garbage_per_ward_32,
    'Distance Region 3 Truck 2' : distance_per_ward_32,
    'Garbage Region 3 Truck 3' : garbage_per_ward_33,
    'Distance Region 3 Truck 3' : distance_per_ward_33,})

temp2.to_csv('Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Complete Data.csv', index = False)

# v1 = [v11, v12, v13]
# v2 = [v21, v22, v23]
# v3 = [v31, v32, v33]

# print('--------------- SAVING STATISTICS ----------------------\n')
# # Save Statistics

# stats = pd.DataFrame(
#     {
#         'Fill Ward 1 (in %)' : [
#             round(gar11, 4), 
#             round(gar12, 4),
#             round(gar13, 4),
#             '-'],
#         'Garbage Fill Ward 1 (in Litres)' : [
#             round(gar11/10 * B_TO_B, 4),
#             round(gar12/10 * B_TO_B, 4),
#             round(gar13/10 * B_TO_B, 4),
#             '-'],
#         'Distance Travelled Ward 1 (in m)' : [
#             round(dist11, 4),
#             round(dist12, 4),
#             round(dist13, 4),
#             '-'],
#         'Garbage per Meter Ward 1 (in KG/m)' : [
#             round(gar11/dist11, 4),
#             round(gar12/dist12, 4),
#             round(gar13/dist13, 4),
#             '-'],
#         'Percentage of Bins covered Ward 1 (in %)' : [
#             round( 100 * (v11.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
#             round( 100 * (v12.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
#             round( 100 * (v13.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
#             round( 100 * np.sum([i.shape[0] - 2 for i in v1])/ data[data.Ward == 0].shape[0], 4)],
#         'Fill Ward 2 (in %)' : [
#             round(gar21, 4), 
#             round(gar22, 4),
#             round(gar23, 4),
#             '-'],
#         'Garbage Fill Ward 2 (in Litres)' : [
#             round(gar21/10 * B_TO_B, 4),
#             round(gar22/10 * B_TO_B, 4),
#             round(gar23/10 * B_TO_B, 4),
#             '-'],
#         'Distance Travelled Ward 2 (in m)' : [
#             round(dist21, 4),
#             round(dist22, 4),
#             round(dist23, 4),
#             '-'],
#         'Garbage per Meter Ward 2 (in KG/m)' : [
#             round(gar21/dist21, 4),
#             round(gar22/dist22, 4),
#             round(gar23/dist23, 4),
#             '-'],
#         'Percentage of Bins covered Ward 2 (in %)' : [
#             round( 100 * (v21.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
#             round( 100 * (v22.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
#             round( 100 * (v23.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
#             round( 100 * np.sum([i.shape[0] - 2 for i in v2])/ data[data.Ward == 1].shape[0], 4)],
#         'Fill Ward 3 (in %)' : [
#             round(gar31, 4), 
#             round(gar32, 4),
#             round(gar33, 4),
#             '-'],
#         'Garbage Fill Ward 3 (in Litres)' : [
#             round(gar31/10 * B_TO_B, 4),
#             round(gar32/10 * B_TO_B, 4),
#             round(gar33/10 * B_TO_B, 4),
#             '-'],
#         'Distance Travelled Ward 3 (in m)' : [
#             round(dist31, 4),
#             round(dist32, 4),
#             round(dist33, 4),
#             '-'],
#         'Garbage per Meter Ward 3 (in KG/m)' : [
#             round(gar31/dist31, 4),
#             round(gar32/dist32, 4),
#             round(gar33/dist33, 4),
#             '-'],
#         'Percentage of Bins covered Ward 3 (in %)' : [
#             round( 100 * (v31.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
#             round( 100 * (v32.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
#             round( 100 * (v33.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
#             round( 100 * np.sum([i.shape[0] - 2 for i in v3])/ data[data.Ward == 2].shape[0], 4)],
#     }, index=['Truck 1', 'Truck 2', 'Truck 3', 'Total Percentage'])
# stats.to_csv('Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/Statistics.csv')

# print('--------------- GENERATING MAP ----------------------')
# # Plotting routes

# map = CreateMap()
# map.createRoutes('Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/', N_WARDS, N_TRUCKS, W1, W2, Multiple_truck = True)
# map.createLatLong('Data/Bin Locations.csv', N_WARDS)
# map.createRoutesDict(N_WARDS)
# map.addRoutesToMap(N_WARDS, N_TRUCKS)
# map.addDepot()
# map.addNodes('Data/Bin Locations.csv')
# map.saveMap('Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/')
# map.displayMap('Data/Dynamic Data/Multiple Trucks/3 Trucks Sensitivity/')