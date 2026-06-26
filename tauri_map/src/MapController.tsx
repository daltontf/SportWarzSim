import ReactDOM from "react-dom/client";
import { useEffect, useState } from "react";
import { useMap } from "react-leaflet";

import { type CalculatorInteface, type LeagueStats, type League, type Team } from "./Structs";
import L, { type LeafletMouseEvent } from "leaflet";

function opacityForPopulation(share_population_value: number) {
  if (share_population_value > 5000000) return 0.9;
  if (share_population_value > 1000000) return 0.8;
  if (share_population_value > 500000) return 0.7;
  if (share_population_value > 100000) return 0.6;
  if (share_population_value > 50000) return 0.5;
  if (share_population_value > 10000) return 0.4;
  if (share_population_value > 5000) return 0.3;
  if (share_population_value > 1000) return 0.2;
  return 0.1;
}

function styleMap(calculations: any) {
  return (feature: any) => {
    const county = calculations.county_stats_by_geoid[feature.properties.geoid];
    return {
      color: "grey",
      weight: 1,
      fillColor: county.team_stats?.[0]?.color || "gray",
      fillOpacity: opacityForPopulation(
        county.team_stats?.[0]?.share_population,
      ),
    };
  };
}

interface MapControllerProps {
  calculator: CalculatorInteface;
  calculations?: LeagueStats;
  league?: League;
  updateTeams: (teams: Team[]) => void;
}

function countyMouseOver(event: LeafletMouseEvent) {
  event.target.setStyle({ color: "white" });
}

function countyMouseOut(event: LeafletMouseEvent) {
  event.target.setStyle({ color: "gray" });
}

function TeamEditor({ action, team, deleteFn }: { action: any, deleteFn: any, team: Team }) {
  return (
    <form action={action}>
      <div style={{ display: "grid", width: "10px", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
          <label>Name:</label>
          <input type="text" defaultValue={team.name} name="name" />
          <label>L:</label>
          <input type="number" defaultValue={team.L} name="L" min={0} max={10} step={0.1}/>
          <label>S:</label>
          <input type="number" defaultValue={team.S} name="S" min={0} max={10} step={0.1}/>
          <label>N:</label>
          <input type="number" defaultValue={team.N} name="N" min={0} max={10} step={0.1}/>
          <label>Color:</label>
          <input type="color" defaultValue={team.color} name="color" />
          <label>State:</label>
          <input type="text" defaultValue={team.state} name="state" />
          <input type="hidden" defaultValue={team.coordinates.lat} name="lat" />
          <input type="hidden" defaultValue={team.coordinates.lon} name="lon" />
          <button type="submit">Save</button>
          { deleteFn && <button type="button" onClick={deleteFn}>Delete</button> }
      </div>
    </form>
  );
}

export default function MapController({
  calculator,
  calculations,
  league,
  updateTeams
}: MapControllerProps) {
  const [geojson, setGeoJsonData] = useState<any>(null);

  const map: L.Map = useMap();

  function enableMapDrag() {
    map.dragging.enable();
  }

  function disableMapDrag() {
    map.dragging.disable();
  }

  async function fetchGeoJson() {
    return await fetch("/counties_4326.geojson")
      .then((res) => res.text())
      .then((text) => JSON.parse(text));
  }

  useEffect(() => {
    fetchGeoJson().then(setGeoJsonData);
  }, []);

  useEffect(() => {
    if (!geojson || !calculations) return;

    const geoJsonLayer = L.geoJSON(geojson, {
      style: styleMap(calculations),
      onEachFeature: (feature, layer) => {
        (layer.bindPopup(() => {
          var leagues_rows = "";
          var county =
            calculations.county_stats_by_geoid[feature.properties.geoid];
          if (county?.team_stats) {
            for (const team_stat of county.team_stats) {
              if (team_stat.share > 1 / county.team_stats.length) {
                leagues_rows += `<tr>
                                      <td>${team_stat.team_name}</td>
                                      <td style="text-align: right;">${team_stat.share_population.toLocaleString("en-US", { maximumFractionDigits: 0 })}</td>
                                      <td style="text-align: right;">${team_stat.share_population_value.toLocaleString("en-US", { maximumFractionDigits: 1 })}</td>
                                    </tr>`;
              } else {
                break;
              }
            }
          }
          return `<table style="border-collapse: collapse;">
                                <caption>${feature.properties.name} - Pop: ${county.population.toLocaleString("en-US")}</caption>
                                <tr>
                                  <th>Team</th>
                                  <th>Pop. Share</th>
                                  <th>Pop. Value</th>
                                </tr>${leagues_rows}</table>`;
        }),
          layer.on({
            mouseover: countyMouseOver,
            mouseout: countyMouseOut,
          }));
      },
    }).addTo(map);

    return () => {
      geoJsonLayer.remove();
    };
  }, [geojson, calculations]);

  useEffect(() => {
    if (!league) return;
    let markers = [];
    league.teams.forEach((team: Team, _: number) => {
      if (team.color) {
        const marker: L.Marker = L.marker(
          [team.coordinates.lat, team.coordinates.lon],
          {
            title: team["name"],
            draggable: true,
          },
        ).addTo(map);

        const markerPopupDiv = document.createElement("div");
        const markerRoot = ReactDOM.createRoot(markerPopupDiv);

        const updateTeam = (data: FormData) => {
          team.name = data.get("name") as string;
          team.L = Number(data.get("L"));
          team.S = Number(data.get("S"));
          team.N = Number(data.get("N"));
          team.color = data.get("color") as string;
          team.state = data.get("state") as string;
          map.closePopup();
          updateTeams(league.teams);
        };

        const deleteFn = () => {
          league.teams = league.teams.filter((t) => t !== team);
          updateTeams(league.teams);
        };

        markerRoot.render(<TeamEditor action={updateTeam} deleteFn={deleteFn} team={team} />);
        marker.bindPopup(markerPopupDiv);

        marker.addEventListener("dragstart", disableMapDrag);
        marker.addEventListener("dragend", async (e) => {
          team.coordinates.lat = e.target._latlng.lat;
          team.coordinates.lon = e.target._latlng.lng;

          team.state = await calculator.getStateForCoordinates(
            team.coordinates.lat,
            team.coordinates.lon,
          );
          updateTeams(league.teams);
        });
        enableMapDrag();

        markers.push(marker);
        }
      map.addEventListener("contextmenu", (e) => {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        calculator.getStateForCoordinates(lat, lon).then((state) => {
          const newTeam: Team = {
            name: "New Team",
            L: 1.0,
            S: 0.0,
            N: 0.0,
            color: "#000000",
            coordinates: {
              lat,
              lon,
            },
            state: state,
          };

          const addTeam = (data: FormData) => {
            league.teams.push({
              name: data.get("name") as string,
              L: Number(data.get("L")),
              S: Number(data.get("S")),
              N: Number(data.get("N")),
              color: data.get("color") as string,
              state: data.get("state") as string,
              coordinates: {
                lat: Number(data.get("lat")),
                lon: Number(data.get("lon")),
              },
            });
            map.closePopup();
            updateTeams(league.teams);
          };

          const addTeamPopupDiv = document.createElement("div");
          ReactDOM.createRoot(addTeamPopupDiv).render(<TeamEditor action={addTeam} team={newTeam} deleteFn={undefined} />);

          L.popup().setLatLng([lat, lon]).setContent(addTeamPopupDiv).openOn(map);
        });
      });
    });

    return () => {
      markers.forEach((marker) => {
        marker.remove();
      });
    };
  }, [league]);

  return null;
}
