export interface Item {
  PersonId: number;
  geometry: string;
  lat: number;
  lng: number;
  edge_lat: number;
  edge_lng: number;
  edge_item_dist: number;
  cluster: number;
  centroid_lat: number;
  centroid_lng: number;
  centroid_item_dist: number;
  centroid_edge_lat: number;
  centroid_edge_lng: number;
  centroid_edge_dist: number;
}

export interface Bounds {
  left?: number;
  bottom?: number;
  right?: number;
  top?: number;
}

export interface Circle {
  mode: number;
  lat: number;
  lng: number;
  color: string;
  tooltip: string;
}

export interface Cluster {
  num: number;
  freq: number;
  lat: number;
  lng: number;
  color: string;
}
