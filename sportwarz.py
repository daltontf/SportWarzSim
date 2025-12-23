import json
import shapely
import numpy as np
import pandas as pd
import ipywidgets as widgets
import itables

from typing import TypedDict, Union, Dict, NewType
from IPython.display import display
from ipyleaflet import Map, GeoJSON, Popup, FullScreenControl, basemaps

class Coordinates(TypedDict):
    lat: float
    lon: float

    def __init__(self, lat:float, lon:float):
        self["lat"] = lat
        self["lon"] = lon

class Team(TypedDict, total=False):
    name: str
    venue: str
    L: float
    S: float
    N: float
    color: str
    state: Union[str, list[str]]
    coordinates: Coordinates

class LeagueJson(TypedDict):
    league_name: str
    teams: list[Team]

class League(TypedDict):
    league_name: str
    weight: float
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

def haversine_miles(centroid, lat2, lon2):
    r = 3958.8  # earth radius miles
    lat1 = centroid.y
    lon1 = centroid.x
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2*r*np.arcsin(np.sqrt(a))

def opacity_for_population(share_population_value): 
    if share_population_value > 5000000:
        return 0.9
    if share_population_value > 1000000:
        return 0.8
    if share_population_value > 500000:
        return 0.7  
    if share_population_value > 100000:
        return 0.6 
    if share_population_value > 50000:
        return 0.5
    if share_population_value > 10000:
        return 0.4
    if share_population_value > 5000:
        return 0.3
    if share_population_value > 1000:
        return 0.2
    return 0.1    

def league_teams_sums(league_data) -> pd.DataFrame:
    league_df = pd.concat(league_data["output_dataframe_map"].values())
    return league_df[['team_name', 'share_population', 'share_population_value']]\
            .groupby('team_name')\
            .sum()\
            .sort_values(by='share_population',ascending=False)

class LeaguesModel:
    _counties_geojson: CountiesGEOJson
    _leagues: Leagues = { }
    _co_dataframe: pd.DataFrame  
    _us_median_income: float

    _geojson_layer: GeoJSON = None
    _leaflet_map: Map = None

    competition_temperature_base = 1 # Lower -> winner takes it
    not_nearest_multiplier = 2 # added to distance multiplied (d - nearest_d) 
    not_same_state_multiplier = 2
    canada_multiplier = 2 

    last_click_lat = 37.7
    last_click_lon = -97.3  
    last_click_state: str = "Kansas"

    def load_counties_geojson(self):
        with open("./counties.geojson") as f:
            self._counties_geojson =  json.load(f)
        
    def load_counties_data(self):
        self._co_dataframe = pd.read_csv("./co-est2024-alldata.csv", index_col=["STATE", "COUNTY"], encoding="iso-8859-1")

        co_income_dataframe = pd.read_csv("./Income2023.csv", index_col="FIPS_Code")

        self._us_median_income = co_income_dataframe.loc[0]["Median_Household_Income_2022"]

        for feature in self._counties_geojson["features"]:
            coords_stack = [ feature["geometry"]["coordinates"] ]

            while isinstance(coords_stack[-1][0], list):
                coords_stack.append(coords_stack[-1][0]) 

            coords = coords_stack[-2]    
            try: 
                raw_polygon = shapely.geometry.Polygon(coords) 
                state = feature["properties"]["STATEFP"]
                county = feature["properties"]["COUNTYFP"]
                self._co_dataframe.at[(state, county), "centroid"] = raw_polygon.centroid
                fips = int(f"{state:02d}{county:03d}")
                self._co_dataframe.at[(state, county), "income"] = co_income_dataframe.at[fips, "Median_Household_Income_2022"]
            except Exception as e:
                print(e)
                print("Invalid coordinates:", feature["properties"]["Name"])  

    def read_league(self, league_name: str):
        with open(f"./teams_{league_name}.json") as f:
            return f.read()

    def load_league(self, league_dict)-> str:
        league_name = league_dict["league_name"]
        self._leagues[league_name] = league_dict  
        return league_name                            

    def load_leagues(self, league_names: list[str]):
     for league in league_names:
         with open(f"./teams_{league}.json") as f:
            self._leagues[league] = json.load(f)

    def add_league(self, league_data:League):
        self._leagues = { league_data["league_name"]: league_data }

    def calculate_league_distances(self, league_name):
        teams = self._leagues[league_name]["teams"]
        # compute distance matrix
        d = { }
        for i, c in self._co_dataframe.iterrows():
            d[i] = np.zeros(len(teams))
            if not pd.isna(c["centroid"]):
                for j, t in enumerate(teams):
                    d[i][j] = haversine_miles(c["centroid"], t["coordinates"]["lat"], t["coordinates"]["lon"])
 
        self._leagues[league_name]["distances"] = d       

    def calculate_distances(self):
        for league_name in self._leagues.keys():
            self.calculate_league_distances(league_name) 

    def compute_league_shares(self, league):
        nearest_key = "nearest"

        d = self._leagues[league]["distances"] 
        league_weight = self._leagues[league]["weight"]
        league_distance_decay = .002 / league_weight 

        self._co_dataframe[nearest_key] = float('nan')
        for j, t in enumerate(self._leagues[league]["teams"]):
            for i, c in self._co_dataframe.iterrows():    
                nearest = self._co_dataframe.loc[i, nearest_key]
                if pd.isna(nearest) or d[i][j] < nearest:
                    self._co_dataframe.loc[i, nearest_key] = d[i][j] 

        dataframe_map = { }
        for i, c in self._co_dataframe.iterrows():            
            nearest_d = c[nearest_key]
            nearest_effective_d = 1000000
            county_state = c["STNAME"]
            R = np.zeros_like(d[i])
            distance_value_multipliers = np.zeros_like(d[i])
            for j, t in enumerate(self._leagues[league]["teams"]):
                effective_d = d[i][j]
                if not pd.isna(nearest_d) and effective_d > nearest_d:
                    effective_d = nearest_d + ((effective_d - nearest_d) * self.not_nearest_multiplier)
                team_state = t["state"] 
                if isinstance(team_state, list):
                    if not county_state in team_state:
                        effective_d *= self.not_same_state_multiplier 
                elif county_state != team_state:
                    effective_d *= self.not_same_state_multiplier 
                    if "eh" in t:
                        effective_d *= self.canada_multiplier     
                if effective_d < nearest_effective_d:
                    nearest_effective_d = effective_d 
                N = t.get("N", 5.0)
                team_distance_decay = league_distance_decay * (5/N)
                D = np.exp(-team_distance_decay * min(effective_d, 15000/N))
                DS = np.exp(-team_distance_decay * effective_d * 2)  #short term enthusiasm dissipates faster             
                R[j] = ((league_weight * 10) + t["L"]  * D)  + ((league_weight * 10) + t["S"] * DS)   
                distance_value_multipliers[j] = np.exp(-min(effective_d, 500) / 200) * league_weight

            expR = np.exp(R / self.competition_temperature_base)
          
            shares = expR / expR.sum(keepdims=True)

            dataframe_out = []
            for j, t in enumerate(self._leagues[league]["teams"]):
                share_population = c["POPESTIMATE2020"] * shares[j]
                income_rel = c["income"] / self._us_median_income
                income_mult = income_rel / (income_rel + 0.75)
                share_population_value = share_population\
                    * distance_value_multipliers[j]\
                    * income_mult
                dataframe_out.append({
                        "county": c["CTYNAME"],
                        "state": c["STNAME"],
                        "team_name": t["name"],
                        "color": t["color"],
                        "share": shares[j],
                        "share_population": share_population,
                        "share_population_value": share_population_value
                })
       
            dataframe_map[c.name] = pd.DataFrame(dataframe_out)

        self._leagues[league]["output_dataframe_map"] = dataframe_map
      
    def compute_shares(self):   
        for league in self._leagues.keys(): 
            self.compute_league_shares(league)
                      

    def reset_county_styles(self):
        default_style = {
            "color": "grey",
            "weight": 1,
            "fillColor": "grey",
            "fillOpacity": 0.0,
        }

        for feature in self._counties_geojson["features"]:
            feature["properties"]["style"] = default_style    

    def heatmap_counties(self, league_name: str, share_threshold = 0.01): 
        self.reset_county_styles()
        for feature in self._counties_geojson["features"]:
            state = feature["properties"]["STATEFP"]
            county = feature["properties"]["COUNTYFP"]
            county_rows = self._leagues[league_name]["output_dataframe_map"].get((state, county))
            if not county_rows is None and county_rows.shape[0] > 0:
                group_cols = list(county_rows.columns.difference(["share_population_value"]))
                county_rows = county_rows.groupby(group_cols, as_index=False).max()
                county_row = county_rows.nlargest(1, "share_population_value").iloc[0]
                if county_row["share"] > share_threshold:   
                    feature["properties"]["style"] = {
                        "color": "grey",
                        "weight": 1,
                        "fillColor": county_row["color"] ,
                        "fillOpacity": opacity_for_population(county_row["share_population_value"])
                    }               
    
    def create_show_teams(self, leaflet_map:Map, popup_leagues:Leagues):
        def show_teams(event, feature, **kwargs): 
            statefp = feature["properties"]["STATEFP"]
            countyfp = feature["properties"]["COUNTYFP"]
            row = self._co_dataframe.query(f"STATE == {statefp} & COUNTY == {countyfp}") 
            centroid = row["centroid"].iloc[0]
            population = row["POPESTIMATE2020"].iloc[0]

            all_county_rows = pd.DataFrame()    
            leagues_table = ""  
            try:
                for league in popup_leagues.keys():
                    county_rows = self._leagues[league]["output_dataframe_map"][(statefp, countyfp)][["team_name", 'state', "share_population_value", "share"]]
                    county_rows = county_rows.groupby(['team_name', 'state'], as_index=False).sum() 
                    county_rows["league"] = league
                    all_county_rows = pd.concat([all_county_rows, county_rows], ignore_index=True)
      
                if all_county_rows.empty:
                    return
                all_county_rows = all_county_rows.sort_values(by="share_population_value", ascending=False)
                for i, county_row in all_county_rows.iterrows():
                    if county_row["share"] > 1/len(self._leagues[league]["teams"]):
                        leagues_table += (
                        "<tr>"  
                        f"<td>{county_row['league']}</td>" 
                        f"<td>{county_row['team_name']}</td>" 
                        f"<td>{county_row['share_population_value']:,.0f}</td>" 
                        "</tr>")
       
                self.last_click_lat = centroid.y
                self.last_click_lon = centroid.x
                self.last_click_state = all_county_rows.iloc[0]["state"]

                popup = Popup(location=(centroid.y, centroid.x), 
                    child=widgets.HTML("<table style='border-collapse: collapse;'>" +
                    f'<caption>{feature["properties"]["Name"]} - {population:,.0f}</caption>' +    
                    leagues_table +
                "</table>"  
                ))
            except Exception as e:
                popup = Popup(location=(centroid.y, centroid.x), 
                    child=widgets.HTML(str(e)))
            leaflet_map.add(popup)
        return show_teams  

    def render_map(self, only_league: str) -> Map:
        if only_league: # Only show pop-up for one league
            popup_leagues = { only_league: self._leagues[only_league] } 
        else:
            popup_leagues = self._leagues     

        display(widgets.HTML(
            """
            <style>
            .leaflet-popup-content-wrapper .leaflet-popup-tip {
                background-color: black;
                border: 2px solid black;
            }
            .leaflet-popup-content {
                color: white;
            }
            
            table {
                style="border-collapse: collapse;"
            }

            td {
                padding: 0 10px;
            }
            </style>"""    
        ))

        map = Map(basemap=basemaps.CartoDB.Positron, center=[38.72728229549864, -96.9010842308538], zoom=5, scroll_wheel_zoom=True)

        layer = GeoJSON(data = self._counties_geojson, 
            hover_style = {"fillColor": "white"}
        )
    
        layer.on_click(self.create_show_teams(map, popup_leagues))
    
        map.add(layer)
        map.add(FullScreenControl())
        map.fullscreen = True
        
        self._geojson_layer = layer
        self._leaflet_map = map

        return map   

    def delete_teams(self, league_name: str, teams: list[str]) -> bool:
        deleted = False
        for new_team in teams:
            for i, team in enumerate(self._leagues[league_name]["teams"]):
                if team["name"] == new_team:
                    self._leagues[league_name]["teams"].pop(i)
                    deleted = True
                    break    
        return deleted   

    def refresh_geojson_layer(self):
        if not self._leaflet_map:
            return
        
        if self._geojson_layer:
            self._leaflet_map.remove_layer(self._geojson_layer)
        
        layer = GeoJSON(data = self._counties_geojson, 
            hover_style = {
                "fillColor": "white",
                "fillOpacity": 0.0,
           }
        )   
        layer.on_click(self.create_show_teams(self._leaflet_map, self._leagues)) 
        self._leaflet_map.add(layer)   
        self._geojson_layer = layer   

    def add_teams(self, league_name: str, new_teams: list[Team], recalculate=True):
        self.delete_teams(league_name, list(map(lambda team: team["name"], new_teams)))
        
        self._leagues[league_name]["teams"].extend(new_teams)

        if recalculate:
            self.calculate_distances() 
            self.compute_shares()  

    def update_team(self, league_name: str, team_name: str, attrs: dict[str, object], recalculate=True):
        for i, team in enumerate(self._leagues[league_name]["teams"]):
            if team["name"] == team_name:
                self._leagues[league_name]["teams"][i] = team | attrs
        if recalculate:
            self.calculate_distances() 
            self.compute_shares()  

    def copy_with_just_league(self, league_name: str):
        leagues_model = LeaguesModel()
        leagues_model._co_dataframe = self._co_dataframe
        leagues_model._us_median_income = self._us_median_income
        leagues_model.load_counties_geojson()
        leagues_model._leagues =  { 
            league_name: {           
                "weight": self._leagues[league_name]["weight"],
                "teams":  self._leagues[league_name]["teams"].copy(),
                "output_dataframe_map": self._leagues[league_name]["output_dataframe_map"]
            }
        }
        return leagues_model

    def show_pre_post_merged_results(self, league_name:str, after_model: Leagues):
        pd.set_option('display.float_format', lambda x: '%.0f' % x)
        pd.set_option('display.max_rows', 256)
        pd.set_option('display.width', 256)

        pre_sums = league_teams_sums(self._leagues[league_name])
        post_sums = league_teams_sums(after_model._leagues[league_name])
        merged = pd.merge(pre_sums, post_sums, on='team_name', how='outer', suffixes=("_before", "_after"))
        merged["share_population_before"] = merged["share_population_before"].fillna(0) 
        merged["share_population_after"] = merged["share_population_after"].fillna(0) 
        merged["share_population_value_before"] = merged["share_population_value_before"].fillna(0) 
        merged["share_population_value_after"] = merged["share_population_value_after"].fillna(0) 
        merged["share_population_delta"] = merged["share_population_after"] - merged["share_population_before"]
        merged["share_population_value_delta"] = merged["share_population_value_after"] - merged["share_population_value_before"]
        
        total = merged.agg(['sum']).rename(index={'sum': 'Total'})
 
        full = pd.concat([merged, total])

        with pd.option_context("display.float_format", "{:,.0f}".format):
            return itables.show(full, paging=False, pageLength=100 )