import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
from show_routes import CreateMap

# Some Constants
B_TO_T = 10
B_TO_B = 100 
N_TRUCKS = 6
N_WARDS = 3
W1 = 0.9
W2 = 0.1
STARTNODE = 0
FOLDER = 'Data/Static Data'



def updateData(data : pd.DataFrame, distance : pd.DataFrame, truck_number : int) -> pd.DataFrame:
    '''
    This function adds fill ratio at the start of each run to the database or copies old values to the new column based on truck number it is dealing with
    '''
    if 'fill_ratio_0' in data.columns.to_list():
        # If the fill ratio is already generated once
        fill = data.loc[:, 'fill_ratio_0']
    else:
        fill = np.random.rand(data.shape[0]).tolist() 

    data.insert(data.shape[1], f'fill_ratio_{truck_number}', fill) # This will be updated if the bin is collected
    data.insert(data.shape[1], f'fill_ratio_{truck_number}_{truck_number}', fill) # This is for reference

    # Update Distance
    if 'distance_from_0' not in data.columns.to_list():
        # If it does not exists, add it
        distance_col = distance.iloc[0, data.index.to_list()].to_list()
        data.insert(data.shape[1], 'distance_from_0', distance_col)

    # Add Fill Per Meter column
    fpm_values = data.loc[:, 'fill_ratio_' + str(truck_number)] / data.loc[:, 'distance_from_0']
    data.insert(data.shape[1], f'fill_p_m_{truck_number}', fpm_values)


    return data


def optimize(data : pd.DataFrame, distance : pd.DataFrame, fills : pd.DataFrame, w1 : float, w2 : float, visited : list, truck_number : int) -> tuple:
    '''
    This function does the optimization for each case.
    '''
    mdl = Model('CVRP')
    N = []
    
    for i in data.index.tolist():
        if (i not in visited) and ( data.loc[i, 'fill_ratio_' + str(truck_number)] + sum(data.loc[N, 'fill_ratio_' + str(truck_number)]) ) * B_TO_T <= 100:
            N.append(i)
            visited.append(i)
    V = N + [0]
    A = [(p, q) for p in V for q in V if p != q]
    C = {(p, q) : distance.iloc[p, q] for p,q in A}
    X = mdl.addVars(A, vtype = GRB.BINARY)
    Y = mdl.addVars(V, vtype = GRB.BINARY)
    U = mdl.addVars(N, vtype = GRB.CONTINUOUS)
    obj = quicksum( 
            (w1 * X[p, q] * C[(p, q)]) - (w2 * Y[p] * fills.loc[p, 'fill'] * B_TO_T) for p, q in A
            )
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(obj)
    # Constraints

    mdl.addConstrs(
        quicksum( X[i, j] for j in V if j != i ) == 1 for i in N
    )
    mdl.addConstrs(
        quicksum( X[i, j] for i in V if i != j ) == 1 for j in N
    )
    mdl.addConstr(
        quicksum( Y[i] * fills.loc[i, 'fill'] * B_TO_T for i in N ) <= ( 100)
    )
    mdl.addConstr(
        quicksum( X[STARTNODE, j] for j in N) == 1
    )
    mdl.addConstr(
        quicksum( X[j, STARTNODE] for j in N ) == 1
    )
    mdl.addConstrs(
        (X[i, j] == 1) >> (U[i] + fills.loc[j, 'fill'] * B_TO_T == U[j]) for i,j in A if i != 0 and j != 0
    )

    mdl.addConstrs(
        U[i] >= (fills.loc[i, 'fill'] * B_TO_T) for i in N
    )
    mdl.addConstrs(U[i] <= (100) for i in N)

    # Model Restrictions

    mdl.Params.MIPGap = 0.1
    mdl.Params.TIMELimit = 900

    # Optimization
    mdl.optimize()
    active_arcs = [a for a in A if X[a].x > 0.99]
    return data, visited, active_arcs 


def staticOptimization(data : pd.DataFrame, distance : pd.DataFrame, nt : int, folder : str, ward : str, w1 : float, w2 : float) -> tuple:
    '''
    This is the function which does some pre-processing before optimzation. It calls the optimization function in itself, so no need to call the optimization function.
    '''
    # Renew seed
    np.random.seed(42)

    visits = []
    visited = []
    for truck in range(nt):
        if len(visited) != data.shape[0]:
            visit = pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')})
            data = updateData(data, distance, truck)
            data = data.sort_values(by = 'fill_p_m_' + str(truck), ascending = False)
            fills = pd.DataFrame(
            {'fill' : data.loc[:, 'fill_ratio_' + str(truck)].tolist() + [0.0]}, index = data.index.tolist() + [0]
            )
            data, visited, active_acrs = optimize(data, distance, fills, w1, w2, visited, truck)
            # Data Recording
            next_element = next(
                y for x, y in active_acrs if x == STARTNODE
            )
            while next_element != STARTNODE:
                visit.loc[len(visit)] = [
                    int(next_element), 
                    data.loc[next_element, 'fill_ratio_' + str(truck)]
                    ]
                data.loc[
                    int(next_element), 
                    [
                        'fill_ratio_' + str(truck), 'fill_p_m_' + str(truck)
                        ]
                    ] = [0.0, 0.0]
                next_element = next(
                    y for x, y in active_acrs if x == next_element
                )
            
            # Now we have travelled whole path except docking at depot
            visit.loc[len(visit)] = [
                next_element,
                np.sum(visit.iloc[:, 1])
            ]

            # Now store the data
            # ----------------------------------------
            print(f'Optimization done for truck {truck}')
            # ----------------------------------------

            file_name = f'{folder}/Visited Truck {ward}/visited_truck{ward}_{truck}_{w1}_{w2}.csv'
            visit.to_csv(file_name, index = False)

            visits.append(visit)

        else:
            print(f"\nAll Nodes have been visited already. Truck {truck + 1} is not used.\n")
    
    file_name = f'{folder}/Truck {ward} Data/truck{ward}_{w1}_{w2}.csv'
    data.to_csv(file_name, index = False)

    return data, visits
        


if __name__ == "__main__":
    data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
    distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
    for i in range(distance.shape[0]):
        distance.iloc[:, i] = distance.iloc[:, i]/np.max(distance.iloc[:, i])

    dist = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1) # This is used for logging data

    stats = {}

    for nt in range(1, N_TRUCKS + 1):
        print(f"\n----------------- Number of Trucks : {nt}. -----------------\n")

        

        # WARD 1 OPTIMIZATION
        data1 = data[data.Ward == 0]
        data1, visits1 = staticOptimization(data1, distance, nt, f"{FOLDER}/{nt} Truck", '1', W1, W2)
        print('\n Ward 1 Done \n')

        # WARD 2 OPTIMIZATION
        data2 = data[data.Ward == 1]
        data2, visits2 = staticOptimization(data2, distance, nt, f"{FOLDER}/{nt} Truck", '2', W1, W2)
        print('\n Ward 2 Done \n')

        # WARD 3 OPTIMIZATION
        data3 = data[data.Ward == 2]
        data3, visits3 = staticOptimization(data3, distance, nt, f"{FOLDER}/{nt} Truck", '3', W1, W2)
        print('\n Ward 3 Done \n')

        # Logging statistics

        ward = 1
        temp5 = {}
        # for p, d, b, g in [(path1, dist1, bins1, gar1), (path2, dist2, bins2, gar2), (path3, dist3, bins3, gar3)]:
        for vis, dat in [(visits1, data1), (visits2, data2), (visits3, data3)]:
            temp1 = [] # To temporary store distance
            temp2 = [] # To temporary store garbage
            temp3 = [] # To temporary store paths
            temp4 = [] # To temporary store bins

            for v in vis:
                # for the current truck in the ward 1, 2 or 3.
                temp = [] # To temporary store current path
                bi = 0 # To calculate Number of bins filled
                di = 0 # To calculate distance for the current truck
                for i in range(len(v) - 1):
                    temp.append((int(v.iloc[i, 0]), int(v.iloc[i + 1, 0])))
                    di += dist.iloc[(int(v.iloc[i, 0]), int(v.iloc[i + 1, 0]))]
                temp1.append(di)
                temp2.append(v.iloc[-1, 1] * 10) 
                temp3.append(temp)
                temp4.append(len(v) - 2)
            temp5[f'Fill Percentage in Ward {ward}'] = [round(gar, 4) for gar in temp2] + [None] * (N_TRUCKS + 1 - len(temp2))
            temp5[f'Garbage Fill in Ward {ward}'] = [round(gar, 4) * B_TO_B / B_TO_T for gar in temp2] + [None] * (N_TRUCKS - len(temp2)) + [np.sum([round(gar, 4) * B_TO_B / B_TO_T for gar in temp2])]
            temp5[f'Distance Travelled in Ward {ward}'] = [round(dis, 4) / 1000 for dis in temp1] + [None] * (N_TRUCKS - len(temp1)) + [np.sum([round(dis, 4) / 1000 for dis in temp1])]
            temp5[f'Garbage per meter in Ward {ward}'] = (np.array([round(gar, 4) * B_TO_B / B_TO_T for gar in temp2]) / np.array([round(dis, 4) / 1000 for dis in temp1])).tolist() + [None] * (N_TRUCKS + 1 - len(temp2))
            temp5[f'Bins covered in Ward {ward}'] = temp4 + [None] * (N_TRUCKS - len(temp4)) + [np.sum(temp4)]
            temp5[f'Bins covered percentage in Ward {ward}'] = (np.array(temp4) * 100 / len(dat)).tolist() + [None] * (N_TRUCKS - len(temp4)) + [np.sum(temp4) * 100 / len(dat)]
            ward += 1

        stats[f'{nt} Trucks'] = temp5
        s = pd.DataFrame(stats[f'{nt} Trucks'])
        s.to_csv(f'{FOLDER}/{nt} Truck/statistics.csv', index = False)

        



        


                



        


