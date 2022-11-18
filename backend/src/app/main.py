import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopy.distance import geodesic
from shapely import wkt
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def get_path(name):
    """
    Get absolute path of a data file.
    """
    d = os.path.dirname(__file__)
    p = os.path.join(d, '..', '..', 'data', name)
    f = os.path.abspath(p)
    return f


def read_data():
    """
    Load datatest into a pandas DataFrame.
    """
    file = get_path('input.xlsx')
    df = pd.read_excel(file)
    return df


def pre_process_data(df):
    """
    Remove rows that have null values and create a GeoPandas DataFrame.
    """
    df = df.dropna()
    loc = df['Location']
    geo = loc.apply(wkt.loads)
    gfd = gpd.GeoDataFrame(df, geometry=geo)
    gfd['lng'] = gfd.geometry.x
    gfd['lat'] = gfd.geometry.y
    return gfd


def find_regions(gdf):
    """
    Calculate boundaries of the location data and split them into squares of
    1x1 kilometer. Total number of items in each region is indicated by a
    population column.
    """
    left, bottom, right, top = gdf.total_bounds
    center_x = (left + right) / 2
    center_y = (bottom + top) / 2
    width = geodesic((center_y, left), (center_y, right))
    height = geodesic((bottom, center_x), (top, center_x))
    rows = int(width.km)
    cols = int(height.km)
    reg_width = abs(right - left) / rows
    reg_height = abs(top - bottom) / cols
    regs = dict()
    for _, item in gdf.iterrows():
        lng = item['lng']
        lat = item['lat']
        row = int((lng - left) / reg_width)
        col = int((lat - bottom) / reg_height)
        num = item['reg_num'] = row * cols + col
        reg = regs.get(num)
        if not reg:
            reg_lng = left + row * reg_width + reg_width / 2
            reg_lat = bottom + col * reg_height + reg_height / 2
            reg = regs[num] = {
                'num': num,
                'row': row,
                'col': col,
                'width': reg_width,
                'height': reg_height,
                'geometry': Point(reg_lng, reg_lat),
                'lng': reg_lng,
                'lat': reg_lat,
                'population': 0
            }
        reg['population'] += 1
    vals = regs.values()
    return gpd.GeoDataFrame(vals)


def normalize_data(gdf):
    """
    Since longitude and latitude have different scales we should normalize them
    between 0 and 1.
    """
    scaler = MinMaxScaler()
    x = gdf[['lng', 'lat']]
    y = scaler.fit_transform(x)
    gdf['lng_norm'] = y[:, 0]
    gdf['lat_norm'] = y[:, 1]
    return gdf


def find_clusters(gdf, k):
    """
    Clustering the normalized locations by the K-Means algorithm.
    """
    x = gdf[['lng_norm', 'lat_norm']]
    km = KMeans(k)
    km.fit(x)
    return km.labels_, km.inertia_


def plot_metrics(gdf, max_k):
    """
    The Elbow Method.
    Calculate the Sum of Squared Error for each cluster size and then plot them,
    in order to find the best possible size for clustering.
    """
    metrics = dict()
    for k in range(1, max_k + 1):
        _, metric = find_clusters(gdf, k)
        metrics[k] = metric
        print(f'K: {k}, Metric: {metric}')
    x = metrics.keys()
    y = metrics.values()
    plt.figure()
    plt.xlabel('K')
    plt.ylabel('Metric')
    plt.plot(x, y)
    plt.show()


def prepare_data():
    """
    Make data ready to process.
    """
    print('Preparing data...')
    df = read_data()
    print('Pre-processing data...')
    gdf = pre_process_data(df)
    print('Finding regions...')
    regs = find_regions(gdf)
    print('Normalizing data...')
    regs = normalize_data(regs)
    return regs


def parse_arg(index, d_type, default):
    """
    Read arguments from the command line.
    """
    if index < len(sys.argv) - 1:
        arg = sys.argv[index + 1]
        if d_type:
            return d_type(arg)
        else:
            return arg
    else:
        return default


def show_metrics():
    """
    Show the metrics plot.
    """
    gdf = prepare_data()
    print('Calculating metrics...')
    plot_metrics(gdf, 50)


def export_data():
    """
    Cluster the data and save the result in the data folder.
    """
    k = parse_arg(0, int, 10)
    if k < 1:
        print('Invalid cluster size!')
        exit(1)
    gdf = prepare_data()
    print('Finding clusters...')
    clusters, _ = find_clusters(gdf, k)
    gdf['cluster'] = clusters
    print(gdf)
    xlsx = get_path('output.xlsx')
    gdf.to_excel(xlsx, index=False)
    print('Output:', xlsx)


def run():
    export_data()
