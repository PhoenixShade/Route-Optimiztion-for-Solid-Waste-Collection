import pandas as pd
# import numpy as np
import os
import folium
from branca.element import Figure
import json
import openrouteservice as ors
import time
# import webbrowser

# from dynamic_weighted_1_truck import N_TRUCKS

# class CreateMap():
#     '''
#     This class is used to create map with routes that the truck followed.

#     The order of execution should be:
#     1. Creation of object
#     2. Calling function createRoutes()
#     3. Calling function createLatLong()
#     4. Calling function createRoutesDict()
#     5. Calling function addRoutesToMap()
#     6. Calling function addDepot()
#     7. Calling function addNodes()
#     8. Calling function saveMap()
#     9. Calling function displayMap() if you want to display the map, not necessary
#     '''    
#     def __init__(self):
#         '''
#         Inintalization function
#         '''
#         self.LOCATION = [30.741482, 76.768066]
#         self.START_ZOOM = 11.3
#         self.ROUTE_COLORS = [
#             ['#49382d',
#             '#5a40e8',
#             '#26ef8e',
#             '#2d9e3e',
#             '#2c4f31',
#             '#ff5eb9',
#             '#262625'],
#             ['#275468',
#             '#12e5b7',
#             '#b29f33',
#             '#54510f',
#             '#191c11',
#             '#a36595',
#             '#a89c7a'],
#             ['#586b22',
#             '#525a5b',
#             '#3d3344',
#             '#09721a',
#             '#724e5a',
#             '#011a6d',
#             '#ce6a37']]
#         self.WARD_COLORS = ['red', 'green', 'darkblue']
#         self.DEPOT_COLOR = 'black'
#         self.ors_key = "5b3ce3597851110001cf624890d3df4ec58741129d23e67c5af0e306"
#         self.client = ors.Client(key = self.ors_key)
#         self.routes_ward1 = {}
#         self.routes_ward2 = {}
#         self.routes_ward3 = {}
#         self.latlong_coord_dict_ward1 = {}
#         self.latlong_coord_dict_ward2 = {}
#         self.latlong_coord_dict_ward3 = {}
#         self.longlat_coord_dict_ward1 = {}
#         self.longlat_coord_dict_ward2 = {}
#         self.longlat_coord_dict_ward3 = {}
#         self.routes_dict_ward1 = {}
#         self.routes_dict_ward2 = {}
#         self.routes_dict_ward3 = {}
#         self.routes_map = []
#         self.routes_all = {}
#         self.map = folium.Map(location = self.LOCATION, zoom_start = self.START_ZOOM)

#     def createRoutes(self, path : str, nWards : int, nTrucks : int, w1 : float, w2 : float, Multiple_truck : bool = False):
#         '''
#         It create routes_all dictionary

#         Input:
#             1. path : path to the case data (string)
#             2. nWards : Number of wards (integer)
#             3. nTrucks : Number of trucks (integer)
#             4. w1 : Weight w1 (float)
#             5. w2 : Weight w2 (float)
#             6. Multiple_truck : If the case has multiple trucks (Boolean) (Default : False)
#         '''
#         for i in range(nWards):
#             self.routes = {}
#             for j in range(nTrucks):
#                 if Multiple_truck and i == 2 and j == 6:
#                     continue
#                 elif Multiple_truck:
#                     path_open = path + 'Visited Truck ' + str(i + 1) + '/visited_truck' + str(i + 1) + '_' + str(j + 1) + '_' + str(w1) + '_' + str(w2) + '.csv'
#                 else:
#                     path_open = path + 'Visited Truck ' + str(i + 1) + '/visited_truck' + str(i + 1) + '_' + str(w1) + '_' + str(w2) + '.csv'
#                 arcs = pd.read_csv(path_open)
#                 self.routes[j] = arcs.Node.astype('int').tolist()
#             self.routes_all[i] = self.routes
    
#     def createLatLong(self, path : str, nWards : int):
#         '''
#         It creates latlong list

#         Input : 
#             1. path : path to data containing nodes latitude and longtitude (string)
#             2. nWards : number of wards (integer)
#         '''
#         nodes = pd.read_csv(path, index_col = 'id')
#         self.latlong = [
#             [self.latlong_coord_dict_ward1, self.longlat_coord_dict_ward1],
#             [self.latlong_coord_dict_ward2, self.longlat_coord_dict_ward2],
#             [self.latlong_coord_dict_ward3, self.longlat_coord_dict_ward3]]
#         for i in range(1, nWards + 1):
#             for key, value in self.routes_all[i - 1].items():
#                 latlong_path = []
#                 longlat_path = []
#                 for step in value:
#                     latlong_path.append([nodes.loc[step, 'y'], nodes.loc[step, 'x']])
#                     longlat_path.append([nodes.loc[step, 'x'], nodes.loc[step, 'y']])
#                 self.latlong[i - 1][0][key] = latlong_path
#                 self.latlong[i - 1][1][key] = longlat_path
    
#     def createRoutesDict(self, nWards : int):
#         '''
#         It creates routes_dict_all dictonary

#         Input : 
#             1. nWards : number of wards (integer)
#         '''
#         self.routes_dict_all = [
#             self.routes_dict_ward1, 
#             self.routes_dict_ward2, 
#             self.routes_dict_ward3]
        
#         for i in range(1, nWards + 1):
#             for key, value in self.latlong[i - 1][0].items():
#                 time.sleep(0.5)
#                 direction_api = self.client.directions(
#                     coordinates = self.latlong[i - 1][1][key],
#                     profile = 'driving-car',
#                     format = 'geojson'
#                 )
#                 coords = direction_api['features'][0]['geometry']['coordinates']
#                 self.routes_dict_all[i - 1][key] = [[coord[1], coord[0]] for coord in coords]
    
#     def addRoutesToMap(self, nWards : int, nTrucks : int):
#         '''
#         This function add routes to the map

#         Input : 
#             1. nWards : number of wards (integer)
#             2. nTrucks : number of trucks (integer)
#         '''
#         # Ward 1
#         cargo1 = folium.FeatureGroup("Cargo 1")
#         cargo2 = folium.FeatureGroup("Cargo 2")
#         cargo3 = folium.FeatureGroup("Cargo 3")
#         cargo4 = folium.FeatureGroup("Cargo 4")
#         cargo5 = folium.FeatureGroup("Cargo 5")
#         cargo6 = folium.FeatureGroup("Cargo 6")
#         cargo7 = folium.FeatureGroup("Cargo 7")

#         # Ward 2
#         cargo8 = folium.FeatureGroup("Cargo 8")
#         cargo9 = folium.FeatureGroup("Cargo 9")
#         cargo10 = folium.FeatureGroup("Cargo 10")
#         cargo11 = folium.FeatureGroup("Cargo 11")
#         cargo12 = folium.FeatureGroup("Cargo 12")
#         cargo13 = folium.FeatureGroup("Cargo 13")
#         cargo14 = folium.FeatureGroup("Cargo 14")

#         # Ward 2
#         cargo15 = folium.FeatureGroup("Cargo 15")
#         cargo16 = folium.FeatureGroup("Cargo 16")
#         cargo17 = folium.FeatureGroup("Cargo 17")
#         cargo18 = folium.FeatureGroup("Cargo 18")
#         cargo19 = folium.FeatureGroup("Cargo 19")
#         cargo20 = folium.FeatureGroup("Cargo 20")
#         cargo21 = folium.FeatureGroup("Cargo 21")

#         cargos = [
#             [cargo1, cargo2, cargo3, cargo4, cargo5, cargo6, cargo7],
#             [cargo8, cargo9, cargo10, cargo11, cargo12, cargo13, cargo14],
#             [cargo15, cargo16, cargo17, cargo18, cargo19, cargo20, cargo21]
#         ]
#         truck_1, truck_2, truck_3, truck_4, truck_5, truck6, truck7 = None, None, None, None, None, None, None
#         trucks = [truck_1, truck_2, truck_3, truck_4, truck_5, truck6, truck7]
#         for i in range(nWards):
#             for j in range(nTrucks):
#                 if i == 2 and j == 6:
#                     continue
#                 else:
#                     truck_1 = folium.vector_layers.PolyLine(
#                         self.routes_dict_all[i][j],
#                         popup = f'<b>Path of Ward {i + 1} Truck {j + 1}</b>',
#                         tooltip = f'Truck {j + 1}',
#                         color = self.ROUTE_COLORS[i][j],
#                         weight = 4
#                         ).add_to(cargos[i][j])
#                     cargos[i][j].add_to(self.map)
    
#     def addDepot(self):
#         '''
#         This function add depot location to the map
#         '''

#         folium.CircleMarker(
#         location = self.latlong_coord_dict_ward1[0][0],
#         popup = 'Depot',
#         tooltip = '<strong>Click here to see Popup</strong>',
#         color = self.DEPOT_COLOR,
#         radius = 1,
#         fill = True, 
#         fill_color = self.DEPOT_COLOR
#         ).add_to(self.map)
    
#     def addNodes(self, path : str):
#         '''
#         This function adds nodes location to map

#         Input : 
#             1. path : path to bin locations file (string)
#         '''
#         self.nodes = {}
#         bins = pd.read_csv(path, index_col = 'id')
#         for key in np.unique(bins.Ward):
#             if key == -1:
#                 continue
#             y = bins[bins.Ward == key].loc[:, 'y'].tolist()
#             x = bins[bins.Ward == key].loc[:, 'x'].tolist()
#             y_x = []
#             for i in range(len(y)):
#                 y_x.append([y[i], x[i]])
#             self.nodes[key] = y_x

#         #add nodes markers
#         for key, value in self.nodes.items():
#             for step in range(len(value)):
#                 folium.CircleMarker(
#                     radius = 1,
#                     location = self.nodes[key][step],
#                     color = self.WARD_COLORS[key],
#                     fill = True,
#                     fill_color = self.WARD_COLORS[key]
#                     ).add_to(self.map)

#     def saveMap(self, path : str):
#         '''
#         This function saves the map as .html file

#         Input :
#             1. path : path to where the map needs to be saved. It will be better to save it in the same directory as the data (string)
#         '''
#         self.map.save(path + 'routes_map.html')
#         self.map.save(path + 'routes_map.json')

#     def displayMap(self, path : str):
#         '''
#         This function displays the map

#         Input : 
#             1. path : path where the map .html file is saved, it will be similar to the one used for saving (string)
#         '''
#         webbrowser.open(path + 'routes_map.html')


class CreateJSON():
    """This class is used to make the JSON file for routes

    """
    def __init__(self, nWards : int, nTrucks : int, w1 : float, w2 : float) -> None:
        LOCATION = [30.741482, 76.768066]
        START_ZOOM = 11.3
        self.ROUTE_COLORS = [
            ['#49382d',
            '#5a40e8',
            '#26ef8e',
            '#2d9e3e',
            '#2c4f31',
            '#ff5eb9',
            '#262625'],
            ['#275468',
            '#12e5b7',
            '#b29f33',
            '#54510f',
            '#191c11',
            '#a36595',
            '#a89c7a'],
            ['#586b22',
            '#525a5b',
            '#3d3344',
            '#09721a',
            '#724e5a',
            '#011a6d',
            '#ce6a37']]
        self.WARD_COLORS = ['red', 'green', 'darkblue']
        self.DEPOT_COLOR = 'black'
        self.ors_key = "5b3ce3597851110001cf624890d3df4ec58741129d23e67c5af0e306"
        self.client = ors.Client(key = self.ors_key)
        self.routes_ward1 = {}   # Here keys will be truck number
        self.routes_ward2 = {}
        self.routes_ward3 = {}
        self.routes = [self.routes_ward1, self.routes_ward2, self.routes_ward3]
        self.latlong = {}    # Here keys will be node number
        self.map = folium.Map(location = LOCATION, zoom_start = START_ZOOM)
        self.nWards = nWards
        self.nTrucks = nTrucks
        self.w1 = w1
        self.w2 = w2

    
    def setLatLong(self, path : str) -> None:
        nodes = pd.read_csv(path, index_col = 'id')
        for node in nodes.index.tolist():
            self.latlong[node] = (nodes.loc[node, 'x'], nodes.loc[node, 'y'])
    
    def setRoutesDict(self, path : str, multiple_trucks : bool = True) -> None:
        
        for i in range(self.nWards):
            for j in range(self.nTrucks):
                if multiple_trucks and i == 2 and j == 6:
                    continue
                elif multiple_trucks:
                    path_open = path + 'Visited Truck ' + str(i + 1) + '/visited_truck' + str(i + 1) + '_' + str(j + 1) + '_' + str(self.w1) + '_' + str(self.w2) + '.csv'
                else:
                    path_open = path + 'Visited Truck ' + str(i + 1) + '/visited_truck' + str(i + 1) + '_' + str(self.w1) + '_' + str(self.w2) + '.csv'
                arcs = pd.read_csv(path_open)
                self.routes[i][j] = [(self.latlong[int(arcs.loc[p, 'Node'])], self.latlong[int(arcs.loc[p+1, 'Node'])]) for p in range(arcs.shape[0] -1)]
    
    def saveRoute(self, decoded : dict, folder_path : str, truck : int, ward : int) -> None:
        save_path = os.path.join(folder_path, 'Routes', f'Ward {ward}', f'Truck {truck}')
        with open(f'{save_path}', 'w') as f:
            json.dump(decoded, f)
    
    def run(self, nodes_path : str, folder_path : str, multiple_trucks : bool) -> None:
        self.setLatLong(nodes_path)
        self.setRoutesDict(folder_path, multiple_trucks)
        
        for ward in range(len(self.routes)):
            for truck in range(self.nTrucks):
                decoded = {}
                count = 0
                for coords in self.routes[ward][truck]:
                    time.sleep(0.5)
                    line = self.client.directions(coords)['routes'][0]['geometry']
                    decoded[count] = ors.convert.decode_polyline(line)
                    count += 1
                self.saveRoute(decoded, folder_path, truck + 1, ward + 1)
    

