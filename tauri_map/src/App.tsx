import { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import { Tabs, Tab, TabList, TabPanel} from "react-tabs"

import MapController from './MapController';

import "leaflet/dist/leaflet.css";
import "react-tabs/style/react-tabs.css"

export interface CalculatorInteface {
  getCalculationsForLeague: (league:any) => Promise<any>,
  getStateForCoordinates: (lat:number, lon:number) => Promise<string>
}

export default function App({calculator}: any) {
  const [geojson, setGeoJsonData] = useState<any>(null);
  const [calculations, setCalculations] = useState<any>(null);
  const [league, setLeague] = useState<any>(null);
  const [teamFile, setTeamFile] = useState("teams_MLB.json");


  async function fetchGeoJson() {
    return await fetch("/counties-4326.geojson")
        .then((res) => res.text())
        .then((text) => JSON.parse(text));
  }

  async function calculate() { 
    setCalculations(await fetch(teamFile)
       .then((res) => res.text())
       .then((text) => {
          setLeague(JSON.parse(text)["teams"]);
          return calculator.getCalculationsForLeague(text)
       }))
  }

  useEffect(() => {
    fetchGeoJson().then(setGeoJsonData);
  }, []); 

  return (
    <Tabs>
      <TabList>
        <Tab>Map</Tab>
        <Tab>Reports</Tab>
      </TabList>
    <TabPanel>
    <div style={{ height: '100%', width: '100%' }}>
      <MapContainer
        center={[40, -95]}
        zoom={4}
        style={{ height: "92vh", width: "100%" }}
       >        
        <TileLayer
          attribution="&copy; OpenStreetMap contributors &copy; CARTO"
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        <MapController geojson={geojson} calculations={calculations} teams={league}/>
      </MapContainer>
      <div style={{ 
          display: "flex", 
          position: "absolute",
          bottom: "10px",
          left: "10px",
          zIndex: 1000
        }}>
        <label>League:</label>  
        <select value={teamFile} onChange={(e) => setTeamFile(e.target.value)}>
          <option value="teams_MLB.json">MLB</option>
        </select>
        <button onClick={calculate}>Calculate</button>        
      </div>
    </div>
    </TabPanel>
    <TabPanel>
    </TabPanel>
    </Tabs> 
  );
}
