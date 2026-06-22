import { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import L, { type LeafletMouseEvent } from 'leaflet';

function opacityForPopulation(share_population_value: number) { 
    if (share_population_value > 5000000) return 0.9
    if (share_population_value > 1000000) return 0.8
    if (share_population_value > 500000) return 0.7  
    if (share_population_value > 100000) return 0.6 
    if (share_population_value > 50000) return 0.5
    if (share_population_value > 10000) return 0.4
    if (share_population_value > 5000) return 0.3
    if (share_population_value > 1000) return 0.2
    return 0.1  
}

function styleMap(calculations: any) {
    return (feature: any) => {
        const county = calculations.county_stats_by_geoid[feature.properties.geoid];
        return {
            color: "grey",
            weight: 1,
            fillColor: county.team_stats?.[0]?.color ||'gray',
            fillOpacity: opacityForPopulation(county.team_stats?.[0]?.share_population)
        };
    }
}

export interface MapControllerProps {
    geojson: any;
    calculations: any;
    teams?: any;
}


function countyMouseOver(event: LeafletMouseEvent) {
    event.target.setStyle({ color: "white" })
}

function countyMouseOut(event: LeafletMouseEvent) {
    event.target.setStyle({ color: "gray" })
}

export default function MapController({ geojson, calculations, teams }: MapControllerProps) {
    const map: L.Map = useMap();

    function enableMapDrag() {
        map.dragging.enable()   
    }

    function disableMapDrag() {
        map.dragging.disable()   
    }

    useEffect(() => {
        if (!geojson || !calculations) return;

        const geoJsonLayer = L.geoJSON(geojson, {
            style: styleMap(calculations),
            onEachFeature: (feature, layer) => {
                layer.bindPopup(() => {
                    var leagues_rows = ""
                    var county = calculations.county_stats_by_geoid[feature.properties.geoid];
                    if (county?.team_stats) {
                        for (const team_stat of county.team_stats) {
                            if (team_stat.share > 1 / county.team_stats.length) {
                                leagues_rows += `<tr>  
                                      <td>${team_stat.team_name}</td> 
                                      <td style="text-align: right;">${team_stat.share_population.toLocaleString('en-US',{ maximumFractionDigits: 0 })}</td> 
                                      <td style="text-align: right;">${team_stat.share_population_value.toLocaleString('en-US', { maximumFractionDigits: 1 })}</td> 
                                    </tr>`;
                            } else {
                                break;
                            }
                        }
                    }
                    return `<table style="border-collapse: collapse;">
                                <caption>${feature.properties.name} - Pop: ${county.population.toLocaleString('en-US')}</caption>
                                <tr>
                                  <th>Team</th>
                                  <th>Pop. Share</th>
                                  <th>Pop. Value</th>
                                </tr>${leagues_rows}</table>`;
                }),
                layer.on({
                    mouseover: countyMouseOver,
                    mouseout: countyMouseOut
                });
            }   
        }).addTo(map);       

        return () => {
            geoJsonLayer.remove();
        };
    }, [geojson, calculations]);

    useEffect(() => {
        if (!teams) return;
         let markers = [];
         teams.forEach((team: any, _: number) => {
           if (team.color) {
            const marker: L.Marker = L.marker([team.coordinates.lat, team.coordinates.lon], {
                title: team["name"],
                draggable: true,
            }).addTo(map);

            marker.addEventListener("dragstart", disableMapDrag);
            marker.addEventListener("dragend", enableMapDrag);

            markers.push(marker);

            //marker.bindPopup(`<b>${team["name"]}</b><br>League: ${team.league}<br>Division: ${team.division}`);
           }    
        });

        return () => {
            markers.forEach(marker => {
                marker.remove();
            });
        }
    }, [teams])

    return null;
}