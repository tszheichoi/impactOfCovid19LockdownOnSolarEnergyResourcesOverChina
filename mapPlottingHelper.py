from mpl_toolkits.basemap import Basemap
import numpy as np
from plotly.graph_objects import *

m = Basemap(resolution = 'l')

# Make trace-generating function (return a Scatter object)
def make_scatter(x,y, color):
    return Scatter(
        x = x,
        y = y,
        mode = 'lines',
        line_color = color,
        line_width = 2,
        name = ' '  # no name on hover
    )

# Functions converting coastline/country polygons to lon/lat traces
def polygons_to_traces(poly_paths, N_poly, color):
    ''' 
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    traces = []  # init. plotting list 

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify = False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse = True)
        
        # add plot.ly plotting options
        traces.append(make_scatter(lon_cc, lat_cc, color = color))
     
    return traces

# Function generating coastline lon/lat traces
def get_coastline_traces(color = '#293742'):
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 150  # use only the 91st biggest coastlines (i.e. no rivers)
    return polygons_to_traces(poly_paths, N_poly, color = color)

# Function generating country lon/lat traces
def get_country_traces():
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    return polygons_to_traces(poly_paths, N_poly)