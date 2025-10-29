import json
import shapely
import numpy as np
import pandas as pd
import ipywidgets as widgets

from typing import TypedDict, Union
from IPython.display import display
from ipyleaflet import Map, GeoJSON, Popup, FullScreenControl, basemaps
from shapely.geometry import Polygon, Point

class Coordinates(TypedDict):
    lat: float
    lon: float

class Team(TypedDict, total=False):
    name: str
    venue: str
    L: float
    S: float
    state: Union[str, list[str]]
    coordinates: Coordinates

class League(TypedDict):
    league_name: str
    teams: list[Team]

def load_counties_geojson():
    with open("./counties.geojson") as f:
        return json.load(f)

def load_counties_data(counties_geojson):
    co_data_frame = pd.read_csv('./co-est2024-alldata.csv', encoding='iso-8859-1')

    for feature in counties_geojson["features"]:
        coords_stack = [ feature["geometry"]["coordinates"] ]

        while isinstance(coords_stack[-1][0], list):
            coords_stack.append(coords_stack[-1][0]) 

        coords = coords_stack[-2]    
        try: 
            raw_polygon = shapely.geometry.Polygon(coords) 
            state = feature["properties"]["STATEFP"]
            county = feature["properties"]["COUNTYFP"]
            co_data_frame.loc[co_data_frame.query(f"STATE == {state} and  COUNTY == {county}").index, "centroid"] = raw_polygon.centroid
        except Exception as e:
            print(e)
            print("Invalid coordinates:", feature["properties"]["Name"])  
    return co_data_frame

def add_teams_and_distances(leagues, co_data_frame):
    for league in leagues.keys():
        with open(f"./teams_{league}.json") as f:
            leagues[league]["json"]: League = json.load(f)

        teams = pd.DataFrame(leagues[league]["json"]["teams"])

        teams['S'].fillna(0.0, inplace=True)
        teams['L'].fillna(5.0, inplace=True)
       
        # compute distance matrix
        d = np.zeros((len(co_data_frame), len(teams)))
        for i, c in co_data_frame.iterrows():
            if not pd.isna(c["centroid"]):
                for j, t in teams.iterrows():
                    d[i,j] = haversine_miles(c["centroid"], t["coordinates"]['lat'], t["coordinates"]['lon'])


        leagues[league]["teams"] = teams   
        leagues[league]["distances"] = d  


def haversine_miles(centroid, lat2, lon2):
    r = 3958.8  # earth radius miles
    lat1 = centroid.y
    lon1 = centroid.x
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2*r*np.arcsin(np.sqrt(a))

def opacity_for_population(population): 
    if population > 5000000:
        return 0.9
    if population > 2000000:
        return 0.8
    if population > 1000000:
        return 0.7  
    if population > 500000:
        return 0.6 
    if population > 100000:
        return 0.5 
    if population > 50000:
        return 0.4
    if population > 10000:
        return 0.3
    if population > 5000:
        return 0.2
    return 0.1

def compute_shares(leagues, co_data_frame):
    distance_decay = 0.005 
    competition_temperature = 0.5  
    not_nearest_multiplier = 2.0 # added to distance multiplied (d - nearest_d) 
    not_same_state_multiplier = 2.0 # TODO might need bigger multipler if different country, eh
    nearest_key = "nearest"
    
    for league in leagues.keys(): 
        d = leagues[league]["distances"] 
    
        # compute raw R_{c,t}

        R = np.zeros_like(d)
        league_weight = leagues[league]["weight"]
        co_data_frame[nearest_key] = float('nan')
        for j, t in leagues[league]["teams"].iterrows():
            for i, c in co_data_frame.iterrows():
                nearest = co_data_frame.loc[i, nearest_key]
                if pd.isna(nearest) or d[i,j] < nearest:
                    co_data_frame.loc[i, nearest_key] = d[i,j] 
    
        for j, t in leagues[league]["teams"].iterrows():
            for i, c in co_data_frame.iterrows():
                effective_d = d[i,j]
                nearest_d = c[nearest_key]
                if not pd.isna(nearest_d) and effective_d > nearest_d:
                    effective_d = nearest_d + ((effective_d - nearest_d) * not_nearest_multiplier)
                team_state = t["state"] 
                county_state = c["STNAME"]
                if isinstance(team_state, list):
                    if not county_state in team_state:
                        effective_d *= not_same_state_multiplier 
                elif county_state != team_state:
                    effective_d *= not_same_state_multiplier      
                team_distance_decay = distance_decay  # TODO?
                D = np.exp(-team_distance_decay * effective_d) 
                R[i,j] = (league_weight + (t["L"] + t["S"]) / 5) * D   

        expR = np.exp(R / competition_temperature)
        leagues[league]["shares"] =  expR / expR.sum(axis=1, keepdims=True)  

def compute_output_dataframes(leagues, co_data_frame):
    for league in leagues.keys(): 
        d = leagues[league]["distances"]
    
        shares = leagues[league]["shares"]
    
        dataframe_out = []
        for i, c in co_data_frame.iterrows():
            for j, t in leagues[league]["teams"].iterrows():
                if shares[i,j] > 0.035:
                    dataframe_out.append({
                        'county': c['CTYNAME'],
                        'countyfp': c['COUNTY'],
                        'state': c['STNAME'],
                        'statefp': c['STATE'],
                        'team_name': t['name'],
                        'population': c['POPESTIMATE2020'],
                        # 'distance_miles': round(d[i,j],1),
                        'share': shares[i,j],
                })
        leagues[league]["dataframes"] = pd.DataFrame(dataframe_out)    

def reset_county_styles(counties_geojson):
    default_style = {
        "color": "grey",
        "weight": 1,
        "fillColor": "grey",
        "fillOpacity": 0.1,
    }

    for feature in counties_geojson["features"]:
        feature["properties"]["style"] = default_style    

def heatmap_counties(leagues, counties_geojson, league_name, team_color_map, share_theshold = 0.05): 
    for feature in counties_geojson["features"]:
       state = feature["properties"]["STATEFP"]
       county = feature["properties"]["COUNTYFP"]
       county_rows = leagues[league_name]["dataframes"].query(f"statefp == {state} & countyfp == {county}").sort_values(by="share", ascending=False)
       if county_rows.shape[0] > 0:
           group_cols = list(county_rows.columns.difference(['share']))
           county_rows = county_rows.groupby(group_cols, as_index=False).max().sort_values(by="share", ascending=False)
           county_row = county_rows.iloc[0]
           if isinstance(team_color_map, dict) and county_row["team_name"] in team_color_map.keys() and county_row['share'] > share_theshold:   
                feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": team_color_map[county_row["team_name"]],
                "fillOpacity": opacity_for_population(county_row["population"] * county_row["share"])
                }               
           elif isinstance(team_color_map, str):
               feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": team_color_map,
                "fillOpacity": opacity_for_population(county_row["population"])
               }    

def render_map(leagues, counties_geojson, co_data_frame):
    def show_teams(event, feature, **kwargs): 
        statefp = feature["properties"]["STATEFP"]
        countyfp = feature["properties"]["COUNTYFP"]
        row = co_data_frame.query(f"STATE == {statefp} & COUNTY == {countyfp}") 
        centroid = row["centroid"].iloc[0]
        population = row["POPESTIMATE2020"].iloc[0]

        all_county_rows = pd.DataFrame()    
        leagues_table = ""  
        try:
            for league in leagues.keys():
                county_rows = leagues[league]["dataframes"].query(f"statefp == {statefp} & countyfp == {countyfp}")
                group_cols = list(county_rows.columns.difference(['share']))
                county_rows = county_rows.groupby(group_cols, as_index=False).max()
                county_rows["league"] = league
                all_county_rows = pd.concat([all_county_rows, county_rows], ignore_index=True)
      

            all_county_rows = all_county_rows.sort_values(by="share", ascending=False)
            for i, county_row in all_county_rows.iterrows():
                if county_row['share'] > 0.05:
                    leagues_table += (
                        "<tr>"  
                        f"<td>{county_row['league']}</td>" 
                        f"<td>{county_row['team_name']}</td>" 
                        f"<td>{round(county_row['share'] * 100, 1)}%</td>" 
                        # f"<td>{county_row['fans']}</td>" 
                        "</tr>")
       
            popup = Popup(location=(centroid.y, centroid.x), 
                  child=widgets.HTML("<table border='1' style='border-collapse: collapse'>" +
                  f'<caption>{feature["properties"]["Name"]} - {population}</caption>' +    
                  leagues_table +
                  "</table>"  
            ))
        except Exception as e:
            popup = Popup(location=(centroid.y, centroid.x), 
                  child=widgets.HTML(str(e)))
        map.add(popup)    

    display(widgets.HTML(
        """
        <style>
        .leaflet-popup-content-wrapper {
            background-color: black;
            border: 2px solid black;
        }
        .leaflet-popup-content {
            color: white;
        }
        </style>"""    
    ))

    map = Map(basemap=basemaps.CartoDB.Positron, center=[38.72728229549864, -96.9010842308538], zoom=5, scroll_wheel_zoom=True)

    layer = GeoJSON(data = counties_geojson, 
        hover_style = {"fillColor": "white"}
    )
    
    layer.on_click(show_teams)
    
    map.add(layer)
    map.add(FullScreenControl())
    map.fullscreen = True
    return map                                