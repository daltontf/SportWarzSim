import ReactDOM from 'react-dom/client';
import App from '../../../tauri_map/src/App';
import {type CalculatorInteface} from "../../../tauri_map/src/Structs";

import buildUrl from 'build-url-ts'

import "../../../tauri_map/src/setupLeaflet";
import '../../../tauri_map/src/index.css';

const restUrl = import.meta.env.VITE_REST_CALCULATOR_ENDPOINT;

let calculator: CalculatorInteface = {
    getCalculationsForLeague: async (league:any) => {
    return fetch(buildUrl(restUrl, {
      queryParams: {
        competition_temperature_base: 1.0
      }
    }),{
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(league)
    }).then((res) => res.json())  
    },
    getStateForCoordinates: async (lat:number, lon:number) => {
      return fetch(restUrl + `?lat=${lat}&lon=${lon}`, { method: 'GET' }).then((res) => res.text())         
    }
}

ReactDOM.createRoot(
  document.getElementById('root')!
).render(
//   <React.StrictMode>
    <App calculator = {calculator}/>
//   </React.StrictMode>
);
