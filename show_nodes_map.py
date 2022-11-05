import pandas as pd
import numpy as np
import folium
from branca.element import Figure
import json
import openrouteservice as ors
import time
import webbrowser

LOCATION = [30.741482, 76.768066]
START_ZOOM = 11.3
nWards = 3
path = 'Data/Bin Locations.csv'
# nTrucks = 
map = folium.Map(location = LOCATION, zoom_start = START_ZOOM)
WARD_COLORS = {-1 : 'black', 0 : 'red', 1 : 'green', 2 : 'darkblue'}
DEPOT_COLOR = 'black'
ors_key = "5b3ce3597851110001cf624890d3df4ec58741129d23e67c5af0e306"
client = ors.Client(key = ors_key)
nodes = {}
bins = pd.read_csv(path, index_col = 'id')
for key in np.unique(bins.Ward):
    y = bins[bins.Ward == key].loc[:, 'y'].tolist()
    x = bins[bins.Ward == key].loc[:, 'x'].tolist()
    y_x = []
    for i in range(len(y)):
        y_x.append([y[i], x[i]])
    nodes[key] = y_x

#add nodes markers
for key, value in nodes.items():
    for step in range(len(value)):
        folium.CircleMarker(
            radius = 5,
            location = nodes[key][step],
            color = WARD_COLORS[key],
            fill = True,
            fill_color = WARD_COLORS[key],
            fill_opacity = 1.0
            ).add_to(map)

map.save('Images/nodes_map.html')
webbrowser.open('Images/nodes_map.html')