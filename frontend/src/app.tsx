import {Spin} from 'antd';
import axios from 'axios';
import _ from 'lodash';
import React, {FC, useMemo, useState} from 'react';
import * as XLSX from 'xlsx';
import {Neshan, NeshanContext, NeshanOptions} from './neshan';

const COLORS = [
  '#e6194B',
  '#3cb44b',
  '#ffe119',
  '#4363d8',
  '#f58231',
  '#911eb4',
  '#42d4f4',
  '#f032e6',
  '#bfef45',
  '#fabed4',
  '#469990',
  '#dcbeff',
  '#9A6324',
  '#fffac8',
  '#800000',
  '#aaffc3',
  '#808000',
  '#ffd8b1',
  '#000075',
  '#a9a9a9',
  '#ffffff',
  '#000000'
];

async function readData(): Promise<Item[]> {
  const res = await axios.get('output.xlsx', {responseType: 'arraybuffer'});
  const book = XLSX.read(res.data);
  const name = book.SheetNames[0];
  const sheet = book.Sheets[name];
  return XLSX.utils.sheet_to_json<Item>(sheet);
}

export const App: FC = () => {

  const colors = useMemo(() => _.shuffle(COLORS), []);
  const options = useMemo<NeshanOptions>(() => ({}), []);
  const [clusters, setClusters] = useState<[number, number][] | null>(null);

  async function onMapInit(context: NeshanContext): Promise<void> {
    const data = await readData();
    let bounds: { left?: number, bottom?: number, right?: number, top?: number } = {};
    let _clusters: [number, number][] = [];
    for (const item of data) {
      const cluster = _.find(_clusters, ['[0]', item.cluster]);
      if (cluster)
        cluster[1] += item.population;
      else
        _clusters.push([item.cluster, item.population]);
      const rect = context.leaflet.rectangle(
        [
          [item.lat - item.height / 2, item.lng - item.width / 2],
          [item.lat + item.height / 2, item.lng + item.width / 2]
        ],
        {
          color: colors[item.cluster % colors.length],
          fillOpacity: 0.75,
          weight: 0.5
        }
      );
      rect.addTo(context.map);
      rect.on('mouseover', () => {
        rect.bindTooltip(`#${item.cluster + 1} (${item.population})`, {direction: 'top'});
        rect.openTooltip();
      });
      rect.on('mouseout', () => {
        rect.unbindTooltip();
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
    context.map.fitBounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]]);
    _clusters = _.sortBy(_clusters, '[0]');
    setClusters(_clusters);
  }

  return (
    <div className="relative w-screen h-screen bg-gray-100">
      <div className="h-full p-8 flex space-x-8">
        <Neshan
          className="flex-1 h-full border border-solid border-gray-300 rounded-lg z-0"
          options={options}
          onInit={onMapInit}/>
        <div
          className="w-48 h-full p-2 flex flex-col space-y-2 border border-solid border-gray-300 rounded-lg overflow-auto">
          <h3 className="text-center">Clusters</h3>
          {clusters?.map(([cluster, population]) => (
            <div
              key={cluster}
              className="p-4 flex items-center justify-center text-white rounded cursor-pointer"
              style={{backgroundColor: colors[cluster]}}>
              <span style={{textShadow: '0px 1px 2px rgba(0, 0, 0, 0.25)'}}>Cluster #{cluster + 1} ({population})</span>
            </div>
          ))}
        </div>
      </div>
      {!clusters && (
        <div className="absolute inset-0 h-full flex items-center justify-center bg-gray-100">
          <Spin size="large"/>
        </div>
      )}
    </div>
  );

};

interface Item {
  num: number;
  row: number;
  col: number;
  width: number;
  height: number;
  geometry: string;
  lat: number;
  lng: number;
  population: number;
  lat_norm: number;
  lng_norm: number;
  cluster: number;
}
