import ReactDOM from 'react-dom/client';
import App from './App';
import {type CalculatorInteface} from "./App";
import { invoke } from '@tauri-apps/api/core';
import "./setupLeaflet";
import './index.css';


let calculator: CalculatorInteface = {
    getCalculationsForLeague: async (league:any) => {
      return invoke('load_league', { leagueData: league })
    },
    getStateForCoordinates: async (lat:number, lon:number) => {
      return Promise.resolve("TODO")
    }
}

ReactDOM.createRoot(
  document.getElementById('root')!
).render(
//   <React.StrictMode>
    // <App/>
    <App calculator = {calculator}/>
//   </React.StrictMode>
);
