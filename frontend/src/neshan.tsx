import _ from 'lodash';
import React, {FC, HTMLAttributes, useEffect, useMemo, useRef} from 'react';

declare const L: NeshanLeaflet;

export const API_KEY = 'web.77208647fddc45dfabb5f9a83b638366';

export const DEFAULT_OPTIONS: NeshanOptions = {
  key: API_KEY,
  maptype: 'dreamy',
  center: [35.699739, 51.338097],
  traffic: false,
  poi: true,
  zoom: 14
};

export const Neshan: FC<NeshanProps> = props => {

  const container = useRef<HTMLDivElement>(null);
  const attrs = useMemo(() => _.omit(props, ['options', 'onInit']), [props]);

  useEffect(() => {
    const map = new L.Map(container.current, {...DEFAULT_OPTIONS, ...props.options});
    props.onInit?.({leaflet: L, map});
    return () => map.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.options]);

  return (
    <div ref={container} {...attrs}></div>
  );

};

export interface NeshanProps extends HTMLAttributes<HTMLDivElement> {
  options?: NeshanOptions;
  onInit?: (context: NeshanContext) => void;
}

export interface NeshanContext {
  leaflet: NeshanLeaflet;
  map: NeshanMap;
}

export type NeshanOptions = any;

export type NeshanLeaflet = any;

export type NeshanMap = any;
