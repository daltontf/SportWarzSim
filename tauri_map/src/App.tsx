import { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import { Tabs, Tab, TabList, TabPanel} from "react-tabs"

import MapController from './MapController';
import { type LeagueStats, type League, type Team } from './Structs';

import "leaflet/dist/leaflet.css";
import "react-tabs/style/react-tabs.css"
import CalculationsComparision from './CalculationsComparision';

export default function App({calculator}: any) {
  const [teamFile, setTeamFile] = useState("");
  const [calculations, setCalculations] = useState<LeagueStats | null>(null);
  const [priorCalculations, setPriorCalculations] = useState<LeagueStats | null>(null);
  const [league, setLeague] = useState<League | null>(null);

  useEffect(() => {
    if (!teamFile) return;
     fetch(teamFile)
      .then((res) => res.text())
      .then((text) => setLeague(JSON.parse(text)))
  }, [teamFile])

  const calculateRef = useRef<HTMLButtonElement>(null);  

  async function calculate() {
    calculateRef.current.disabled = true;
    setCalculations(await calculator.getCalculationsForLeague(league))
    calculateRef.current.disabled = false;
  }   

  function updateTeams(teams: Team[]) {
    setLeague((prevLeague) => ({ ...prevLeague, teams }));
  }  

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
        <MapController calculator={calculator} calculations={calculations} league={league} updateTeams={updateTeams} />
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
          <option value="" disabled>-</option>
          <option value="teams_MLB.json">MLB</option>
          <option value="teams_MLS.json">MLS</option>
          <option value="teams_NBA.json">NBA</option>
          <option value="teams_NFL.json">NFL</option>
          <option value="teams_NHL.json">NHL</option>
        </select>
        <button ref={calculateRef} onClick={calculate}>Calculate</button>        
      </div>
    </div>
    </TabPanel>
    <TabPanel>
      <CalculationsComparision calculations={calculations} priorCalculations={priorCalculations}/>
    </TabPanel>
    </Tabs> 
  );
}
