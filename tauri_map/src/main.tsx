import ReactDOM from 'react-dom/client';

import App from './App';
import {type CalculatorInteface} from "./Structs";
import { invoke } from '@tauri-apps/api/core';
import "./setupLeaflet";
import './index.css';

let calculator: CalculatorInteface = {
    getCalculationsForLeague: async (league:any) => {
      return invoke('load_league', { league })
    },
    getStateForCoordinates: async (lat:number, lon:number) => {
      return invoke('lookup_state_name_by_coordinates', { lat, lon })
    }
}

ReactDOM.createRoot(
  document.getElementById('root')!
).render(<App calculator = {calculator}/>);
