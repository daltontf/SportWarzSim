import json
import shapely
import numpy as np
import pandas as pd
import ipywidgets as widgets

from typing import TypedDict, Union, Dict, NewType
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
    color: str
    state: Union[str, list[str]]
    coordinates: Coordinates

class LeagueJson(TypedDict):
    league_name: str
    teams: list[Team]

class League(TypedDict):
    json: LeagueJson
    dataframe: pd.DataFrame
    distances: np.ndarray
    shares: np.ndarray

Leagues = NewType("Leagues", Dict[str, League])

class Geometry(TypedDict):
    type: str
    coordinates: list[object]

class Feature(TypedDict):
    properties: dict[str, object] 
    geometry: Geometry

class CountiesGEOJson(TypedDict):
    features: list[Feature]

def load_counties_geojson() -> CountiesGEOJson:
    with open("./counties.geojson") as f:
        return json.load(f)

def load_counties_data(counties_geojson: CountiesGEOJson):
    co_data_frame = pd.read_csv("./co-est2024-alldata.csv", encoding="iso-8859-1")
    
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

def load_leagues(leagues: Leagues):
     for league in leagues.keys():
         with open(f"./teams_{league}.json") as f:
            leagues[league]["json"]: LeagueJson = json.load(f)


def calculate_distances(leagues: Leagues, co_data_frame: pd.DataFrame):
    for league in leagues.keys():
        teams = leagues[league]["json"]["teams"]

        # compute distance matrix
        d = np.zeros((len(co_data_frame), len(teams)))
        for i, c in co_data_frame.iterrows():
            if not pd.isna(c["centroid"]):
                for j, t in enumerate(teams):
                    d[i,j] = haversine_miles(c["centroid"], t["coordinates"]["lat"], t["coordinates"]["lon"])
 
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
    if population > 1000000:
        return 0.8
    if population > 500000:
        return 0.7  
    if population > 100000:
        return 0.6 
    if population > 50000:
        return 0.5
    if population > 10000:
        return 0.4
    if population > 5000:
        return 0.3
    if population > 1000:
        return 0.2
    return 0.1

def compute_shares(leagues: Leagues, co_data_frame: pd.DataFrame):
    competition_temperature_base = 1 # Lower -> winner takes it
    not_nearest_multiplier = 2 # added to distance multiplied (d - nearest_d) 
    not_same_state_multiplier = 2.5
    canada_multiplier = 2 
    nearest_key = "nearest"
    
    for league in leagues.keys(): 
        d = leagues[league]["distances"] 
        league_weight = leagues[league]['json']["weight"]
        league_distance_decay = .002 / league_weight 

        shares = np.zeros_like(d)
        co_data_frame[nearest_key] = float('nan')
        for j, t in enumerate(leagues[league]["json"]["teams"]):
            for i, c in co_data_frame.iterrows():    
                nearest = co_data_frame.loc[i, nearest_key]
                if pd.isna(nearest) or d[i,j] < nearest:
                    co_data_frame.loc[i, nearest_key] = d[i,j] 

        for i, c in co_data_frame.iterrows():            
            nearest_d = c[nearest_key]
            nearest_effective_d = 1000000
            county_state = c["STNAME"]
            R = np.zeros_like(d[i])
            for j, t in enumerate(leagues[league]["json"]["teams"]):
                effective_d = d[i,j]
                N = t.get("N", 5.0)
                if not pd.isna(nearest_d) and effective_d > nearest_d:
                    effective_d = nearest_d + ((effective_d - nearest_d) * not_nearest_multiplier)
                team_state = t["state"] 
                if isinstance(team_state, list):
                    if not county_state in team_state:
                        effective_d *= not_same_state_multiplier 
                elif county_state != team_state:
                    effective_d *= not_same_state_multiplier 
                    if "eh" in t:
                        effective_d *= canada_multiplier     
                if effective_d < nearest_effective_d:
                    nearest_effective_d = effective_d 
                team_distance_decay = league_distance_decay * (5/N)
                D = np.exp(-team_distance_decay * min(effective_d, 15000/N))
                DS = np.exp(-team_distance_decay * effective_d * 2)  #short term enthusiasm dissipates faster             
                R[j] = ((league_weight * 10) + t["L"]  * D)  + ((league_weight * 10) + t["S"] * DS)   

            #competition_temperature = np.log10((5 + d.min()) / 5) + competition_temperature_base
            expR = np.exp(R / competition_temperature_base)
          
            raw_shares =  expR / expR.sum(keepdims=True)

            # Another attempt at simulating "non-fandom"
            # min_share = raw_shares.min() + (.01 * raw_shares.max())
            # for j in range(len(raw_shares)):
            #     raw_shares[j] = max(0, raw_shares[j] - min_share)

            shares[i] = raw_shares

        leagues[league]["shares"] = shares     

def compute_output_dataframes(leagues: Leagues, co_data_frame: pd.DataFrame):
    for league in leagues.keys(): 
        d = leagues[league]["distances"]
    
        shares = leagues[league]["shares"]
    
        dataframe_out = []
        for i, c in co_data_frame.iterrows():
            for j, t in enumerate(leagues[league]["json"]["teams"]):
                share = shares[i,j]
                share_population = c["POPESTIMATE2020"] * share
                #if share > 0.001: # or share_population > 1000: # algoritm assumes everyone is a fan of some team in then
                dataframe_out.append({
                        "county": c["CTYNAME"],
                        "countyfp": c["COUNTY"],
                        "state": c["STNAME"],
                        "statefp": c["STATE"],
                        "team_name": t["name"],
                        "color": t["color"],
                        "share_population": share_population,
                        "effective_distance": d[i,j],
                        "share": share,
                })
        leagues[league]["dataframes"] = pd.DataFrame(dataframe_out)    

def reset_county_styles(counties_geojson: CountiesGEOJson):
    default_style = {
        "color": "grey",
        "weight": 1,
        "fillColor": "grey",
        "fillOpacity": 0.0,
    }

    for feature in counties_geojson["features"]:
        feature["properties"]["style"] = default_style    

def heatmap_counties(leagues: Leagues, counties_geojson:CountiesGEOJson, league_name: str, share_threshold = 0.05): 
    for feature in counties_geojson["features"]:
       state = feature["properties"]["STATEFP"]
       county = feature["properties"]["COUNTYFP"]
       county_rows = leagues[league_name]["dataframes"].query(f"statefp == {state} & countyfp == {county}").sort_values(by="share", ascending=False)
       if county_rows.shape[0] > 0:
           group_cols = list(county_rows.columns.difference(["share"]))
           county_rows = county_rows.groupby(group_cols, as_index=False).max().sort_values(by="share", ascending=False)
           county_row = county_rows.iloc[0]
           if county_row["share"] > share_threshold:   
                feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": county_row["color"] ,
                "fillOpacity": opacity_for_population(county_row["share_population"])
                }               
        #    elif isinstance(team_color_map, str):
        #        feature["properties"]["style"] = {
        #         "color": "grey",
        #         "weight": 1,
        #         "fillColor": team_color_map,
        #         "fillOpacity": opacity_for_population(county_row["population"])
        #        }    

def render_map(leagues: Leagues, only_league: str, counties_geojson:CountiesGEOJson, co_data_frame:pd.DataFrame):
    if only_league: # Only show pop-up for one league
        leagues = { only_league: leagues[only_league] }

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
                county_rows = leagues[league]["dataframes"].query(f"statefp == {statefp} & countyfp == {countyfp}")[["team_name", "share"]]
                county_rows = county_rows.groupby(['team_name'], as_index=False).max()
                county_rows["league"] = league
                all_county_rows = pd.concat([all_county_rows, county_rows], ignore_index=True)
      

            all_county_rows = all_county_rows.sort_values(by="share", ascending=False)
            for i, county_row in all_county_rows.iterrows():
                if county_row["share"] > 1/len(leagues[league]["json"]["teams"]):
                    leagues_table += (
                        "<tr>"  
                        f"<td>{county_row['league']}</td>" 
                        f"<td>{county_row['team_name']}</td>" 
                        f"<td>{round(county_row['share'] * 100, 1)}%</td>" 
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

def league_teams_sums(league_data: League):
    league_dfs = league_data["dataframes"]
    num_teams = len(league_data["json"]["teams"])
    return league_dfs[league_dfs["share"].apply(lambda x: x > 1/num_teams)]\
        [['team_name', 'share_population']]\
        .groupby('team_name')\
        .sum()\
        .sort_values(by='share_population',ascending=False)

def add_team(co_data_frame: pd.DataFrame, league_name: str, new_teams: list[str]):
    #Don't mutate exisiing data
    leagues_singular = { league_name: { }}
    load_leagues(leagues_singular)
       
    leagues_singular[league_name]["json"]["teams"].extend(new_teams)

    calculate_distances(leagues_singular, co_data_frame) 
    compute_shares(leagues_singular, co_data_frame)  
    compute_output_dataframes(leagues_singular, co_data_frame)    
    return leagues_singular

def update_team(co_data_frame: pd.DataFrame, league_name: str, team_name: str, attrs: dict[str, object]):
    leagues_singular = { league_name: { }}
    load_leagues(leagues_singular)

    for i, team in enumerate(leagues_singular[league_name]["json"]["teams"]):
        if team["name"] == team_name:
            leagues_singular[league_name]["json"]["teams"][i] = team | attrs
    calculate_distances(leagues_singular, co_data_frame) 
    compute_shares(leagues_singular, co_data_frame)  
    compute_output_dataframes(leagues_singular, co_data_frame)  
    return leagues_singular