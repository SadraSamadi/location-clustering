import argparse
import os
import warnings
from time import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.colors import CSS4_COLORS
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import nearest_points
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def get_path(name):
    """
    Get absolute path of a data file.
    """
    d = os.path.dirname(__file__)
    p = os.path.join(d, '..', '..', 'data', name)
    f = os.path.abspath(p)
    return f


def load_data(name, frac):
    """
    Read data from the input file, find samples by fraction,
    remove unused info, and convert to GeoPandas dataframe.
    """
    print('loading data')
    file = get_path(name)
    df = pd.read_excel(file)
    df = df.dropna()
    df = df.sample(frac=frac, ignore_index=True, random_state=0)
    df = df.drop(['Latitude', 'Longitude'], axis=1)
    loc = df.pop('Location')
    geo = loc.apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=geo)
    gdf['lat'] = gdf.geometry.y
    gdf['lng'] = gdf.geometry.x
    print('data shape:', gdf.shape)
    return gdf


def download_map(data, file):
    """
    Download the OpenStreetMap graph for data bounds.
    """
    print('downloading map')
    left, bottom, right, top = data.total_bounds
    filters = '["highway"~"motorway|trunk|primary|secondary"]'
    graph = ox.graph_from_bbox(top, bottom, right, left, custom_filter=filters)
    ox.save_graph_xml(graph, file)
    return graph


def load_map(data, name, dl):
    """
    Read/Download the OpenStreetMap graph file.
    """
    print('loading map graph')
    file = get_path(name)
    graph = download_map(data, file) if dl else ox.graph_from_xml(file)
    edges = ox.graph_to_gdfs(graph, nodes=False)
    print('edges shape:', edges.shape)
    return graph, edges


def haversine_matrix(x, y):
    """
    Calculate the distance matrix of multiple locations by the Haversine formula (km).
    input format: [[lat, lng], ...]
    """
    x_rad = np.deg2rad(x)
    y_rad = np.deg2rad(y)
    h = haversine_distances(x_rad, y_rad)
    return 6371.0 * h  # multiply by radius of the Earth


def haversine_distance(x, y):
    """
    Calculate the distance of two locations by the Haversine formula (km).
    input format: [lat, lng]
    """
    xx = np.array(x, dtype=float, ndmin=2)
    yy = np.array(y, dtype=float, ndmin=2)
    m = haversine_matrix(xx, yy)
    return m[0, 0]


def find_nearest_edges(graph, edges, lat, lng):
    """
    Find the coordinates and distance of the nearest edge to each item (km).
    output: [[lat, lng, dist], ...]
    """
    print('finding nearest edges')
    nearest = ox.nearest_edges(graph, lng, lat)
    result = list()
    for i, n in enumerate(nearest):
        edge = edges.loc[n]
        item = Point(lng[i], lat[i])
        point, _ = nearest_points(edge.geometry, item)
        d = haversine_distance([point.y, point.x], [item.y, item.x])
        result.append([point.y, point.x, d])
    return result


def pre_process_data(data, graph, edges, validate):
    """
    Validate data by dropping the outlier items (the nearest edge > 1km).
    Find the nearest edges to each item.
    """
    print('pre-processing data')
    nearest = find_nearest_edges(graph, edges, data['lat'], data['lng'])
    data[['edge_item_lat', 'edge_item_lng', 'edge_item_dist']] = nearest
    if validate:
        valid = data['edge_item_dist'] <= 1
        invalid = data.shape[0] - valid.sum()
        data = data[valid]
        data = data.reset_index(drop=True)
        print('invalid items dropped:', invalid)
    return data


# noinspection DuplicatedCode
def affinity_propagation(data, param):
    """
    Find clusters by the AffinityPropagation algorithm.
    input: distance matrix in km
    distance function: haversine in km
    centroid preference: nearest distance of each item from its edge
    param: factor for centroid preferences (effects on cluster size)
    output: clusters + centroids
    """
    from sklearn.cluster import AffinityPropagation
    print('affinity propagation')
    alpha = 1000.0 if param is None else param
    print('preference factor:', alpha)
    dist = data['edge_item_dist']
    prefs = -1 * alpha * dist  # inverse to prioritize lower values
    x = data[['lat', 'lng']]
    matrix = -1 * haversine_matrix(x, x)  # inverse to prioritize lower values
    model = AffinityPropagation(preference=prefs, affinity='precomputed', random_state=0)
    model.fit(matrix)
    data['cluster'] = model.labels_
    for i, row in data.iterrows():
        cluster = row['cluster']
        center = model.cluster_centers_indices_[cluster]
        data.loc[i, 'centroid_lat'] = data.loc[center, 'lat']
        data.loc[i, 'centroid_lng'] = data.loc[center, 'lng']
    return data


# noinspection DuplicatedCode
def k_means(data, param):
    """
    Find clusters by the K-Means algorithm.
    input: normalized data coordinates
    param: number of clusters
    output: clusters + centroids
    """
    from sklearn.cluster import KMeans
    print('k-means')
    k = 30 if param is None else int(param)
    print('number of clusters:', k)
    scaler = MinMaxScaler()
    x = data[['lat', 'lng']]
    x_norm = scaler.fit_transform(x)
    model = KMeans(n_clusters=k)
    model.fit(x_norm)
    print('some of squared error:', model.inertia_)
    data['cluster'] = model.labels_
    center = scaler.inverse_transform(model.cluster_centers_)
    for i, row in data.iterrows():
        cluster = row['cluster']
        data.loc[i, ['centroid_lat', 'centroid_lng']] = center[cluster]
    return data


# noinspection DuplicatedCode
def mean_shift(data, param):
    """
    Find clusters by the Mean-Shift algorithm.
    input: normalized data coordinates
    param: bandwidth (size of the region to search through)
    output: clusters + centroids
    """
    from sklearn.cluster import MeanShift
    print('mean shift')
    bw = 0.05 if param is None else param
    print('bandwidth:', bw)
    scaler = MinMaxScaler()
    x = data[['lat', 'lng']]
    x_norm = scaler.fit_transform(x)
    model = MeanShift(bandwidth=bw)
    model.fit(x_norm)
    data['cluster'] = model.labels_
    center = scaler.inverse_transform(model.cluster_centers_)
    for i, row in data.iterrows():
        cluster = row['cluster']
        data.loc[i, ['centroid_lat', 'centroid_lng']] = center[cluster]
    return data


# noinspection DuplicatedCode
def agglomerative_clustering(data, param):
    """
    Find clusters by the Agglomerative Clustering algorithm.
    input: distance matrix in km
    distance function: haversine in km
    param: maximum radius of cluster in km
    output: clusters
    """
    from sklearn.cluster import AgglomerativeClustering
    print('agglomerative clustering')
    t = 1.0 if param is None else param
    print('distance threshold:', t)
    x = data[['lat', 'lng']]
    matrix = haversine_matrix(x, x)
    model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=t)
    model.fit(matrix)
    data['cluster'] = model.labels_
    data[['centroid_lat', 'centroid_lng']] = data[['lat', 'lng']]
    return data


# noinspection DuplicatedCode
def dbscan(data, param):
    """
    Find clusters by the DBSCAN algorithm.
    input: distance matrix in km
    distance function: haversine in km
    param: minimum distance of clusters in km
    output: clusters
    """
    from sklearn.cluster import DBSCAN
    print('dbscan')
    e = 0.5 if param is None else param
    print('epsilon:', e)
    x = data[['lat', 'lng']]
    matrix = haversine_matrix(x, x)
    model = DBSCAN(eps=e, min_samples=1, metric='precomputed')
    model.fit(matrix)
    data['cluster'] = model.labels_
    data[['centroid_lat', 'centroid_lng']] = data[['lat', 'lng']]
    return data


# noinspection DuplicatedCode
def birch(data, param):
    """
    Find clusters by the BIRCH algorithm.
    input: normalized data coordinates
    param: maximum radius of cluster
    output: clusters + centroids
    """
    from sklearn.cluster import Birch
    print('birch')
    t = 0.05 if param is None else param
    print('threshold:', t)
    scaler = MinMaxScaler()
    x = data[['lat', 'lng']]
    x_norm = scaler.fit_transform(x)
    model = Birch(threshold=t, n_clusters=None)
    model.fit(x_norm)
    data['cluster'] = model.labels_
    center = scaler.inverse_transform(model.subcluster_centers_)
    for i, row in data.iterrows():
        cluster = row['cluster']
        data.loc[i, ['centroid_lat', 'centroid_lng']] = center[cluster]
    return data


def find_clusters(data, algorithm, param):
    """
    Clustering data by the specified algorithm.
    """
    print('finding clusters:', end=' ')
    start = time()
    if algorithm == 1:
        data = affinity_propagation(data, param)
    elif algorithm == 2:
        data = k_means(data, param)
    elif algorithm == 3:
        data = mean_shift(data, param)
    elif algorithm == 4:
        data = agglomerative_clustering(data, param)
    elif algorithm == 5:
        data = dbscan(data, param)
    elif algorithm == 6:
        data = birch(data, param)
    finish = time()
    t = round(finish - start, 2)
    print(f'execution time: {t}s')
    return data


def post_process_data(data, graph, edges):
    """
    Calculate the distance of each item to its cluster centroid.
    Find the nearest edges to each centroid.
    """
    print('post-processing data')
    for i, row in data.iterrows():
        item = row[['lat', 'lng']]
        centroid = row[['centroid_lat', 'centroid_lng']]
        data.loc[i, 'centroid_item_dist'] = haversine_distance(centroid, item)
    nearest = find_nearest_edges(graph, edges, data['centroid_lat'], data['centroid_lng'])
    data[['centroid_edge_lat', 'centroid_edge_lng', 'centroid_edge_dist']] = nearest
    return data


def save_data(data, name):
    """
    Write the data to an output file.
    """
    print('saving data')
    file = get_path(name)
    cols = [
        'PersonId',
        'geometry',
        'lat',
        'lng',
        'edge_item_lat',
        'edge_item_lng',
        'edge_item_dist',
        'cluster',
        'centroid_lat',
        'centroid_lng',
        'centroid_item_dist',
        'centroid_edge_lat',
        'centroid_edge_lng',
        'centroid_edge_dist'
    ]
    data.to_excel(file, index=False, columns=cols)
    print('output:', file)


def print_report(data):
    """
    Print out data statistics.
    """
    print('printing reports')
    print('-' * 60)
    print('cluster size (items in each cluster) statistics:')
    cluster = data['cluster']
    cluster_counts = cluster.value_counts()
    cluster_counts_desc = cluster_counts.describe()
    print(cluster_counts_desc)
    print('-' * 60)
    print('centroid-item distance statistics (km):')
    centroid_item_dist = data['centroid_item_dist']
    centroid_item_dist_desc = centroid_item_dist.describe()
    print(centroid_item_dist_desc)
    print('-' * 60)
    print('centroid-edge distance statistics (km):')
    centroid_edge_dist = data['centroid_edge_dist']
    centroid_edge_dist_desc = centroid_edge_dist.describe()
    print(centroid_edge_dist_desc)
    print('-' * 60)


def plot_data(data, edges):
    """
    Show the map graph, clustered data and centroids.
    """
    print('plotting data')
    colors = CSS4_COLORS.values()
    colors = list(colors)
    item_color = list()
    centroid_dict = dict()
    for _, row in data.iterrows():
        cluster = row['cluster']
        color = colors[cluster % len(colors)]
        item_color.append(color)
        if cluster not in centroid_dict:
            centroid_dict[cluster] = {
                'lat': row['centroid_lat'],
                'lng': row['centroid_lng'],
                'color': color
            }
    centroid_vals = centroid_dict.values()
    centroid_df = pd.DataFrame(centroid_vals)
    centroid_lat = centroid_df['lat']
    centroid_lng = centroid_df['lng']
    centroid_color = centroid_df['color']
    base = edges.plot()
    data.plot(ax=base, color=item_color)
    plt.scatter(centroid_lng, centroid_lat, marker='x', s=150, c=centroid_color)
    plt.show()


def parse_args():
    """
    Read arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices={'original', 'njfb'}, default='njfb')
    parser.add_argument('-f', '--fraction', type=float, default=1.0)
    parser.add_argument('-l', '--download', action='store_true')
    parser.add_argument('-v', '--validate', action='store_true')
    parser.add_argument('-m', '--algorithm', type=int, choices={1, 2, 3, 4, 5, 6}, default=1)
    parser.add_argument('-p', '--param', type=float)
    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    data = load_data(f'{args.dataset}/input.xlsx', args.fraction)
    graph, edges = load_map(data, f'{args.dataset}/graph.osm', args.download)
    data = pre_process_data(data, graph, edges, args.validate)
    data = find_clusters(data, args.algorithm, args.param)
    data = post_process_data(data, graph, edges)
    print_report(data)
    save_data(data, f'{args.dataset}/output.xlsx')
    plot_data(data, edges)
