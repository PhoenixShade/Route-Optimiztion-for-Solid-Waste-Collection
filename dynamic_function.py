from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

# Constants
B_TO_T = 10 # Bin to Truck
B_TO_B = 100 # Bin to Bin

def dyn_opt(df1, df2, df3, distances, visit1, visit2, visit3, folder_path, w1 = 0.5, w2 = 0.5, m = 0, n1_done = 0, n2_done = 0, n3_done = 0, obj_value = []):
    SPEED = 13.88
    # Model
    mdl = Model('CVRP')

    visit1.Node = visit1.Node.astype('int')
    visit2.Node = visit2.Node.astype('int')
    visit3.Node = visit3.Node.astype('int')
    

    # Initalization
    obj1, obj2, obj3 = 0, 0, 0 
    f_prev = 'fill_ratio_' + str(m-1)
    fpm = 'fill_per_m_' + str(m)
    f_new = 'fill_ratio_' + str(m)
    start_node = [
        visit1.iloc[-1,0],
        visit2.iloc[-1,0],
        visit3.iloc[-1,0]
        ]
    if n1_done != 1:
        if m == 0:
            fill1 = df1.fill_ratio
        else:
            fill1 = df1.loc[:, f_prev]
        df1.insert(df1.shape[1], f_new, fill1)
        if m != 0:
            for i in df1.index.tolist():
                if i not in visit1.Node.to_list() and np.random.rand() < 0.80:
                    df1.loc[i, f_new] = df1.loc[i, f_new] + np.random.uniform(0, 1 - df1.loc[i, f_new])/10
        dist = distances.iloc[start_node[0], df1.index.to_list()].tolist()
        dist_name = 'distance_from_' + str(start_node[0])
        if dist_name in df1.columns:
            dist_name = dist_name + '_' + str(np.random.rand())
        df1.insert(df1.shape[1], dist_name, dist)
        # df1.insert(df1.shape[1], fpm, B_TO_B*df1.loc[:,f_new]/df1.loc[:,dist_name])
        df1.insert(df1.shape[1], fpm, df1.loc[:,f_new]/df1.loc[:,dist_name])

        df1 = df1.sort_values(by = f_new, ascending = False)
        fill_1 = pd.DataFrame({'fill' : df1.loc[:, f_new].to_list() + [0]}, index = df1.index.tolist() + [0])
        N1 = []
        for i in df1.index.tolist():
            if i not in visit1.Node.to_list() and (df1.loc[i, f_new] + sum(df1.loc[N1, f_new]))*B_TO_T <= 100 - sum(visit1.iloc[:,1])*B_TO_T:
                N1.append(i)
        if m == 0:
            V1 = N1 + [0]
        else:
            V1 = [start_node[0]] + N1 + [0]
        A1 = [(i,j) for i in V1 for j in V1 if i != j]
        c1 = {(i,j) : distances.iloc[i, j] for i,j in A1}
        # c1 = {(i,j) : distances.iloc[i, j]/dis_conv for i,j in A1}
        x1 = mdl.addVars(A1, vtype = GRB.BINARY)
        y1 = mdl.addVars(V1, vtype = GRB.BINARY)
        u1 = mdl.addVars(N1, vtype = GRB.CONTINUOUS)
        obj1 = quicksum( (w1*x1[i, j]*c1[(i, j)]) - w2*y1[i]*fill_1.loc[i, 'fill']*B_TO_T for i,j in A1)


    if n2_done != 1:
        if m == 0:
            fill2 = df2.fill_ratio
        else:
            fill2 = df2.loc[:, f_prev]
        df2.insert(df2.shape[1], f_new, fill2)
        if m != 0:
            for i in df2.index.tolist():
                if i not in visit2.Node.to_list() and np.random.rand() < 0.80:
                    df2.loc[i, f_new] = df2.loc[i, f_new] + np.random.uniform(0, 1 - df2.loc[i, f_new])/10
        dist = distances.iloc[start_node[1], df2.index.to_list()].tolist()
        dist_name = 'distance_from_' + str(start_node[1])
        if dist_name in df2.columns:
            dist_name = dist_name + '_' + str(np.random.rand())
        df2.insert(df2.shape[1], dist_name, dist)
        df2.insert(df2.shape[1], fpm, df2.loc[:,f_new]/df2.loc[:,dist_name])

        df2 = df2.sort_values(by = f_new, ascending = False)
        fill_2 = pd.DataFrame({'fill' : df2.loc[:, f_new].to_list() + [0]}, index = df2.index.tolist() + [0])
        N2 = []
        for i in df2.index.tolist():
            if i not in visit2.Node.to_list() and (df2.loc[i, f_new] + sum(df2.loc[N2, f_new]))*B_TO_T <= 100 - sum(visit2.iloc[:,1])*B_TO_T:
                N2.append(i)
        if m == 0:
            V2 = N2 + [0]
        else:
            V2 = [start_node[1]] + N2 + [0]
        A2 = [(i,j) for i in V2 for j in V2 if i != j]
        c2 = {(i,j) : distances.iloc[i, j] for i,j in A2}
        # c2 = {(i,j) : distances.iloc[i, j]/dis_conv for i,j in A2}
        x2 = mdl.addVars(A2, vtype = GRB.BINARY)
        y2 = mdl.addVars(V2, vtype = GRB.BINARY)
        u2 = mdl.addVars(N2, vtype = GRB.CONTINUOUS)
        obj2 = quicksum( (w1*x2[i, j]*c2[(i, j)]) - w2*y2[i]*fill_2.loc[i, 'fill']*B_TO_T for i,j in A2)
        
    if n3_done != 1:
        if m == 0:
            fill3 = df3.fill_ratio
        else:
            fill3 = df3.loc[:, f_prev]
        df3.insert(df3.shape[1], f_new, fill3)
        if m != 0:
            for i in df3.index.tolist():
                if i not in visit3.Node.to_list() and np.random.rand() < 0.80:
                    df3.loc[i, f_new] = df3.loc[i, f_new] + np.random.uniform(0, 1 - df3.loc[i, f_new])/10
        dist = distances.iloc[start_node[2], df3.index.to_list()].tolist()
        dist_name = 'distance_from_' + str(start_node[2])
        if dist_name in df3.columns:
            dist_name = dist_name + '_' + str(np.random.rand())
        df3.insert(df3.shape[1], dist_name, dist)
        df3.insert(df3.shape[1], fpm, df3.loc[:,f_new]/df3.loc[:,dist_name])
        df3 = df3.sort_values(by = f_new, ascending = False)
        fill_3 = pd.DataFrame({'fill' : df3.loc[:, f_new].to_list() + [0]}, index = df3.index.tolist() + [0])
        N3 = []
        for i in df3.index.tolist():
            if i not in visit3.Node.to_list() and (df3.loc[i, f_new] + sum(df3.loc[N3, f_new]))*B_TO_T <= 100 - sum(visit3.iloc[:,1])*B_TO_T:
                N3.append(i)
        if m == 0:
            V3 = N3 + [0]
        else:
            V3 = [start_node[2]] + N3 + [0]
        A3 = [(i,j) for i in V3 for j in V3 if i != j]
        c3 = {(i,j) : distances.iloc[i, j] for i,j in A3}
        # c3 = {(i,j) : distances.iloc[i, j]/dis_conv for i,j in A3}
        x3 = mdl.addVars(A3, vtype = GRB.BINARY)
        y3 = mdl.addVars(V3, vtype = GRB.BINARY)
        u3 = mdl.addVars(N3, vtype = GRB.CONTINUOUS)
        obj3 = quicksum( (w1*x3[i, j]*c3[(i, j)]) - w2*y3[i]*fill_3.loc[i, 'fill']*B_TO_T for i,j in A3)
   
    # Model
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(obj1 + obj2 + obj3)

    # Constraints
    if n1_done == 0:
        mdl.addConstrs( quicksum( x1[i,j] for j in V1 if j != i) == 1 for i in N1 )
        mdl.addConstrs( quicksum( x1[i,j] for i in V1 if i != j) == 1 for j in N1 )
        mdl.addConstr( quicksum( y1[i]*fill_1.loc[i, 'fill']*B_TO_T for i in N1 ) <= (100 - sum(visit1.iloc[:, 1])*B_TO_T) )
        mdl.addConstr( quicksum( x1[start_node[0], j] for j in N1) == 1)
        if start_node[0] != 0:
            mdl.addConstr( quicksum( x1[j, start_node[0]] for j in N1) == 0)
        mdl.addConstr( quicksum( x1[j, 0] for j in N1) == 1)
        if start_node[0] != 0:
            mdl.addConstr( quicksum( x1[0, j] for j in N1) == 0)
        mdl.addConstrs(
        (x1[i,j] == 1) >> (u1[i] + fill_1.loc[j, 'fill']*B_TO_T == u1[j]) for i,j in A1 if i != 0 and j != 0 and i != int(visit1.iloc[-1,0]) and j != int(visit1.iloc[-1,0]) 
        )
        mdl.addConstrs( u1[i] >= fill_1.loc[i, 'fill']*B_TO_T for i in N1 )
        mdl.addConstrs( u1[i] <= 100 - sum(visit1.iloc[:, 1])*B_TO_T for i in N1 )
    
    if n2_done == 0:
        mdl.addConstrs( quicksum( x2[i,j] for j in V2 if j != i) == 1 for i in N2 )
        mdl.addConstrs( quicksum( x2[i,j] for i in V2 if i != j) == 1 for j in N2 )
        mdl.addConstr( quicksum( y2[i]*fill_2.loc[i, 'fill']*B_TO_T for i in N2 ) <= (100 - sum(visit2.iloc[:, 1])*B_TO_T) )
        mdl.addConstr( quicksum( x2[start_node[1], j] for j in N2) == 1)
        if start_node[1] != 0:
            mdl.addConstr( quicksum( x2[j, start_node[1]] for j in N2) == 0)
        mdl.addConstr( quicksum( x2[j, 0] for j in N2) == 1)
        if start_node[1] != 0:
            mdl.addConstr( quicksum( x2[0, j] for j in N2) == 0)
        mdl.addConstrs(
        (x2[i,j] == 1) >> (u2[i] + fill_2.loc[j, 'fill']*B_TO_T == u2[j]) for i,j in A2 if i != 0 and j != 0 and i != int(visit2.iloc[-1,0]) and j != int(visit2.iloc[-1,0]) 
        )
        mdl.addConstrs( u2[i] >= fill_2.loc[i, 'fill']*B_TO_T for i in N2 )
        mdl.addConstrs( u2[i] <= 100 - sum(visit2.iloc[:, 1])*B_TO_T for i in N2 )
    
    if n3_done == 0:
        mdl.addConstrs( quicksum( x3[i,j] for j in V3 if j != i) == 1 for i in N3 )
        mdl.addConstrs( quicksum( x3[i,j] for i in V3 if i != j) == 1 for j in N3 )
        mdl.addConstr( quicksum( y3[i]*fill_3.loc[i, 'fill']*B_TO_T for i in N3 ) <= (100 - sum(visit3.iloc[:, 1])*B_TO_T) )
        mdl.addConstr( quicksum( x3[start_node[2], j] for j in N3) == 1)
        if start_node[2] != 0:
            mdl.addConstr( quicksum( x3[j, start_node[2]] for j in N3) == 0)
        mdl.addConstr( quicksum( x3[j, 0] for j in N3) == 1)
        if start_node[2] != 0:
            mdl.addConstr( quicksum( x3[0, j] for j in N3) == 0)
        mdl.addConstrs(
        (x3[i,j] == 1) >> (u3[i] + fill_3.loc[j, 'fill']*B_TO_T == u3[j]) for i,j in A3 if i != 0 and j != 0 and i != int(visit3.iloc[-1,0]) and j != int(visit3.iloc[-1,0]) 
        )
        mdl.addConstrs( u3[i] >= fill_3.loc[i, 'fill']*B_TO_T for i in N3 )
        mdl.addConstrs( u3[i] <= 100 - sum(visit3.iloc[:, 1])*B_TO_T for i in N3 )

    # Model time Restrictions

    mdl.Params.MIPGap = 0.1
    mdl.Params.TimeLimit = 900

    # Optimize model
    mdl.optimize()
    q = mdl.getObjective()
    obj_value.append(q.getValue())

    
    # TODO : Time simulation
    if n1_done == 0:
        active_arcs1 = [a for a in A1 if x1[a].x > 0.99]
        TIME = 900 # 15 minutes
        visited1 = 0
        next_element = next( y for x, y in active_arcs1 if x == visit1.iloc[-1, 0] )
        temp = np.max(pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1).iloc[start_node[0], :])
        while (TIME - temp * c1[visit1.iloc[-1, 0], next_element]/(SPEED) >= 0) and next_element != 0:
            TIME = TIME - temp * c1[visit1.iloc[-1, 0], next_element]/(SPEED)
            visit1.loc[len(visit1.index)] = [next_element, df1.loc[next_element, f_new]]
            df1.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs1 if x == visit1.iloc[-1, 0] )
            visited1 = visited1 + 1
        if visited1 == 0:
            print('Forecully entered the value 1.')
            visit1.loc[len(visit1.index)] = [next_element, df1.loc[next_element, f_new]]
            df1.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs1 if x == visit1.iloc[-1, 0] )
        if next_element == 0:
            visit1.loc[len(visit1.index)] = [next_element, sum(visit1.iloc[:, 1])]
            print(f'\nOptimization done for Truck 1')
            file_name = folder_path + 'Truck 1 Data/truck1_' + str(w1) + '_' + str(w2) + '.csv'
            df1.to_csv(file_name, index = False)
            file_name = folder_path + 'Visited Truck 1/visited_truck1_' + str(w1) + '_' + str(w2) + '.csv'
            visit1.to_csv(file_name, index = False)
            n1_done = 1
        print(f'Active arcs | Truck 1 | Start Node : {start_node[0]} :\n{active_arcs1}')

    if n2_done == 0:
        active_arcs2 = [a for a in A2 if x2[a].x > 0.99]
        TIME = 900 # 15 minutes
        visited2 = 0
        next_element = next( y for x, y in active_arcs2 if x == visit2.iloc[-1, 0] )
        temp = np.max(pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1).iloc[start_node[1], :])
        while (TIME - temp * c2[visit2.iloc[-1, 0], next_element]/(SPEED) >= 0) and next_element != 0:
            TIME = TIME - temp * c2[visit2.iloc[-1, 0], next_element]/(SPEED)
            visit2.loc[len(visit2.index)] = [next_element, df2.loc[next_element, f_new]]
            df2.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs2 if x == visit2.iloc[-1, 0] )
            visited2 = visited2 + 1
        if visited2 == 0:
            print('Forecully entered the value 2.')
            visit2.loc[len(visit2.index)] = [next_element, df2.loc[next_element, f_new]]
            df2.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs2 if x == visit2.iloc[-1, 0] )
        if next_element == 0:
            visit2.loc[len(visit2.index)] = [next_element, sum(visit2.iloc[:, 1])]
            print(f'\nOptimization done for truck 2')
            file_name = folder_path + 'Truck 2 Data/truck2_' + str(w1) + '_' + str(w2) + '.csv'
            df2.to_csv(file_name, index = False)
            file_name = folder_path + 'Visited Truck 2/visited_truck2_' + str(w1) + '_' + str(w2) + '.csv'
            visit2.to_csv(file_name, index = False)
            n2_done = 1
        print(f'Active arcs | Truck 2 | Start Node : {start_node[1]} :\n{active_arcs2}')

    if n3_done == 0:
        active_arcs3 = [a for a in A3 if x3[a].x > 0.99]
        TIME = 900 # 15 minutes
        visited3 = 0
        next_element = next( y for x, y in active_arcs3 if x == visit3.iloc[-1, 0] )
        temp = np.max(pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1).iloc[start_node[2], :])
        while (TIME - temp * c3[visit3.iloc[-1, 0], next_element]/(SPEED) >= 0) and next_element != 0:
            TIME = TIME - temp * c3[visit3.iloc[-1, 0], next_element]/(SPEED)
            visit3.loc[len(visit3.index)] = [next_element, df3.loc[next_element, f_new]]
            df3.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs3 if x == visit3.iloc[-1, 0] )
            visited3 = visited3 + 1
        if visited3 == 0:
            print('Forecully entered the value 3.')
            visit3.loc[len(visit3.index)] = [next_element, df3.loc[next_element, f_new]]
            df3.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs3 if x == visit3.iloc[-1, 0] )
        if next_element == 0:
            visit3.loc[len(visit3.index)] = [next_element, sum(visit3.iloc[:, 1])]
            print(f'\nOptimization done for truck 3')
            file_name = folder_path + 'Truck 3 Data/truck3_' + str(w1) + '_' + str(w2) + '.csv'
            df3.to_csv(file_name, index = False)
            file_name = folder_path + 'Visited Truck 3/visited_truck3_' + str(w1) + '_' + str(w2) + '.csv'
            visit3.to_csv(file_name, index = False)
            n3_done = 1
        print(f'Active arcs | Truck 3 | Start Node : {start_node[2]} :\n{active_arcs3}')

    print(f'\n{n1_done, n2_done, n3_done}')
    m = m + 1
    # df4 = pd.concat([df1, df2, df3])
    # df4.loc[0, :] = df.loc[0]
    if n1_done == 1 and n2_done == 1 and n3_done == 1:
        print('\nDone computation')
        return obj_value
    dyn_opt(df1 = df1, df2 = df2, df3 = df3, distances= distances, folder_path= folder_path, visit1 = visit1, visit2 = visit2, visit3 = visit3, m = m, w1 = w1, w2 = w2, n1_done = n1_done, n2_done = n2_done, n3_done = n3_done, obj_value = obj_value)
    return obj_value