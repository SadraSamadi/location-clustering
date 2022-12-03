import {Spin} from 'antd';
import chroma from 'chroma-js';
import cnl from 'color-name-list';
import _ from 'lodash';
import React, {FC, useMemo, useRef, useState} from 'react';
import {readData} from './api';
import {Bounds, Circle, Cluster} from './model';
import {Neshan, NeshanContext, NeshanOptions} from './neshan';

// max cluster radius (km)
const MAX_RADIUS = 1.0;

export const App: FC = () => {

  const colors = useMemo(() => _.shuffle(cnl).map(c => c.hex), []);
  const options = useMemo<NeshanOptions>(() => ({maptype: 'standard-night'}), []);
  const [clusters, setClusters] = useState<Cluster[] | null>(null);
  const ref = useRef<NeshanContext>(null);

  async function onMapInit(context: NeshanContext): Promise<void> {
    ref.current = context;
    const scale = context.leaflet.control.scale({position: 'topright'});
    scale.addTo(context.map);
    await loadData();
  }

  async function loadData(): Promise<void> {
    const data = await readData('output.xlsx');
    let bounds: Bounds = {};
    let _clusters: Cluster[] = [];
    for (const item of data) {
      const color = colors[item.cluster % colors.length];
      let cluster = _.find(_clusters, ['num', item.cluster]);
      if (!cluster) {
        cluster = {
          num: item.cluster,
          freq: 1,
          lat: item.centroid_lat,
          lng: item.centroid_lng,
          color: color
        };
        _clusters.push(cluster);
        createCircle({
          mode: 2,
          lat: cluster.lat,
          lng: cluster.lng,
          color: color,
          tooltip: `Centroid<br/>Cluster #${item.cluster + 1}`
        });
      }
      cluster.freq++;
      createCircle({
        mode: item.centroid_item_dist > MAX_RADIUS ? 0 : 1,
        lat: item.lat,
        lng: item.lng,
        color: color,
        tooltip: `PersonId ${item.PersonId}<br/>Cluster #${item.cluster + 1}`
      });
      if (!bounds.left || item.lng < bounds.left)
        bounds.left = item.lng;
      if (!bounds.bottom || item.lat < bounds.bottom)
        bounds.bottom = item.lat;
      if (!bounds.right || item.lng > bounds.right)
        bounds.right = item.lng;
      if (!bounds.top || item.lat > bounds.top)
        bounds.top = item.lat;
    }
    ref.current.map.fitBounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]]);
    _clusters = _.sortBy(_clusters, 'num');
    setClusters(_clusters);
  }

  function createCircle(config: Circle): void {
    const circle = ref.current.leaflet.circle([config.lat, config.lng], {
      color: _.nth([
        chroma(config.color).brighten().hex(),
        config.color,
        chroma(config.color).darken().hex()
      ], config.mode),
      fillOpacity: _.nth([0.2, 0.4, 0.6], config.mode),
      radius: _.nth([8, 16, 32], config.mode)
    });
    circle.addTo(ref.current.map);
    circle.on('mouseover', () => {
      circle.bindTooltip(config.tooltip, {direction: 'top'});
      circle.openTooltip();
    });
    circle.on('mouseout', () => {
      circle.unbindTooltip();
    });
  }

  function onClusterClicked(cluster: Cluster): void {
    ref.current.map.setView([cluster.lat, cluster.lng], 16);
  }

  return (
    <div className="relative w-screen h-screen bg-gray-900">
      <div className="h-full p-8 flex space-x-8">
        <Neshan
          className="flex-1 h-full border border-solid border-gray-600 rounded-lg z-0"
          options={options}
          onInit={onMapInit}/>
        <div
          className="w-48 h-full p-2 flex flex-col space-y-2 border border-solid border-gray-600 rounded-lg overflow-auto">
          <h3 className="text-center text-white">Clusters ({clusters?.length || 0})</h3>
          {clusters?.map(cluster => (
            <div
              key={cluster.num}
              className="px-4 py-2 flex items-center justify-center text-white transition-opacity duration-300 hover:opacity-75 rounded cursor-pointer"
              style={{backgroundColor: cluster.color}}
              onClick={() => onClusterClicked(cluster)}>
              <span className="text-shadow">Cluster #{cluster.num + 1} ({cluster.freq})</span>
            </div>
          ))}
        </div>
      </div>
      {!clusters && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <Spin size="large"/>
        </div>
      )}
    </div>
  );

};
