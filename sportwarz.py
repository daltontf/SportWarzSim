import json
import sys
from unittest import result
import polars as pl
import ipywidgets as widgets
import itables

from typing import Callable, TypedDict, Union, Dict, NewType
from IPython.display import display
from ipyleaflet import Map, GeoJSON, Popup, FullScreenControl, basemaps

import pyrust

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

class League(TypedDict):
    league_name: str
    teams: list[Team]

League = NewType("League", Dict[str, League])    

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

def league_teams_sums(county_data) -> pl.DataFrame:
    all_teams = pl.DataFrame([x for county in county_data.values() for x in county["team_stats"]])

    return all_teams.select(['team_name', 'share_population', 'share_population_value'])\
            .group_by('team_name')\
            .agg(pl.sum('share_population'), pl.sum('share_population_value'))\
            .sort('share_population_value', descending=True)  

class LeaguesModel:
    _leagues: dict 
    _leagues_calculations: dict
    _counties_geojson: GeoJSON
    _geojson_layer: GeoJSON = None
    _leaflet_map: Map = None

    league_stats_calculator: pyrust.LeagueStatsCalculator

    def __init__(self, leagues_stats_calculator: pyrust.LeagueStatsCalculator = pyrust.LeagueStatsCalculator()):
        self._leagues = {}
        self._leagues_calculations = {}
        self.league_stats_calculator = leagues_stats_calculator
        with open("./counties-4326.geojson") as f:
            self._counties_geojson =  json.load(f) 

    def load_leagues(self, leagues):
        for league in leagues:
            with open(f"teams_{league}.json", "r") as f:
                league = json.load(f)
                self._leagues[league["league_name"]] = league

    def compute_league_shares(self, league_name):
        import os
        endpoint = os.getenv("REST_CALCULATOR_ENDPOINT")
        if endpoint: # USE REST CALCULATOR
            import requests
            response = requests.post(endpoint, json=self._leagues[league_name])
            league_data = response.json()
            league_data["county_stats_by_geoid"] = {int(geoid): value for geoid, value in league_data["county_stats_by_geoid"].items()}
        else: # USE RUST CALCULATOR
            league_data = self.league_stats_calculator.load_league(self._leagues[league_name])

        self._leagues_calculations[league_data["league_name"]] = league_data["county_stats_by_geoid"]

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
        league_county_map = self._leagues_calculations[league_name]
        for feature in self._counties_geojson["features"]:
            geoid = feature["properties"]["geoid"]
            county_data = league_county_map[geoid]
            county_top_team_data = county_data["team_stats"][0]
            if county_top_team_data["share"] > share_threshold:   
                feature["properties"]["style"] = {
                    "color": "grey",
                    "weight": 1,
                    "fillColor": county_top_team_data["color"] ,
                    "fillOpacity": opacity_for_population(county_top_team_data["share_population_value"])
                }                
    
    def create_show_teams(self, leaflet_map:Map, league_name: str):
        league_county_map = self._leagues_calculations[league_name]
        def show_teams(event, feature, **kwargs): 
            coordinates = kwargs.get("coordinates")
            geoid = feature["properties"]["geoid"]
            county_data = league_county_map[geoid]
            county_team_data = county_data["team_stats"]
            leagues_rows = ""  
            for i, county_team_row in enumerate(county_team_data):
                if county_team_row["share"] > 1/len(county_team_data):
                    leagues_rows += f'''
                    <tr>  
                        <td>{county_team_row['team_name']}</td> 
                        <td>{county_team_row['share_population']:,.0f}</td> 
                        <td>{county_team_row['share_population_value']:,.0f}</td> 
                    </tr>'''

                popup = Popup(location = coordinates, max_width=500,
                child=widgets.HTML(f'''
                <table style="border-collapse: collapse;">
                    <caption>{feature["properties"]["name"]} - Pop: {county_data["population"]:,.0f}</caption>
                    <tr>
                        <th>Team</th>
                        <th>Pop. Share</th>
                        <th>Pop. Value</th>
                    </tr>
                {leagues_rows}
                </table>'''))  
            leaflet_map.add(popup)
        return show_teams   

    def render_map(self, only_league: str) -> Map:
        # if only_league: # Only show pop-up for one league
        #     popup_leagues = { only_league: self._leagues[only_league] } 
        # else:
        #     popup_leagues = self._leagues     

        display(widgets.HTML("""
            <style>
            .leaflet-popup-content-wrapper .leaflet-popup-tip {
                background-color: black;
                border: 2px solid black;
            }
            .leaflet-popup-content {
               color: white;
               max-height: 350px;  
               overflow-y: auto; 
            }
            
            table {
                style="border-collapse: collapse;"
            }

            th,td {
                padding: 0 10px;
            }
            </style>"""))

        if only_league:
            self.heatmap_counties(only_league)

        map = Map(basemap=basemaps.CartoDB.Positron, center=[38.72728229549864, -96.9010842308538], zoom=5, scroll_wheel_zoom=True)
        layer = GeoJSON(data = self._counties_geojson, 
            hover_style = {"color": "white"}
        )

        if only_league:
            layer.on_click(self.create_show_teams(map, only_league))

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

    def refresh_geojson_layer(self, only_league: str):
        if not self._leaflet_map:
           return
        
        if self._geojson_layer:
            self._leaflet_map.remove_layer(self._geojson_layer)
        
        layer = GeoJSON(data = self._counties_geojson, 
            hover_style = {"color": "white"}
        )   
        layer.on_click(self.create_show_teams(self._leaflet_map, only_league)) 
        self._leaflet_map.add(layer)   
        self._geojson_layer = layer   

    def add_teams(self, league_name: str, new_teams: list[Team]):
        self.delete_teams(league_name, list(map(lambda team: team["name"], new_teams)))
        self._leagues[league_name]["teams"].extend(new_teams)

    def update_team(self, league_name: str, team_name: str, attrs: dict[str, object]):
        for i, team in enumerate(self._leagues[league_name]["teams"]):
            if team["name"] == team_name:
                self._leagues[league_name]["teams"][i] = team | attrs

    def copy_with_just_league(self, league_name: str):
        leagues_model = LeaguesModel()
        leagues_model._leagues[league_name] = json.loads(json.dumps(self._leagues[league_name]))
        if self._leagues_calculations.get(league_name):
            leagues_model._leagues_calculations[league_name] = json.loads(json.dumps(self._leagues_calculations[league_name]))
        return leagues_model

    def show_pre_post_merged_results(self, league_name:str, after_model):

        before_sums = league_teams_sums(self._leagues_calculations[league_name])\
            .rename({"share_population": "share_population_before", "share_population_value": "share_population_value_before"})

        after_sums = league_teams_sums(after_model._leagues_calculations[league_name])\
            .rename({"share_population": "share_population_after", "share_population_value": "share_population_value_after"})

        merged = before_sums.join(after_sums, on='team_name', how='outer', coalesce=True)\
            .fill_null(0).with_columns([
               (pl.col("share_population_after") - pl.col("share_population_before")).alias("share_population_change"),
               (pl.col("share_population_value_after") - pl.col("share_population_value_before")).alias("share_population_value_change")
            ])
        
        formatter = lambda x: f"{x:,.0f}"
        
        total = merged.select([
              pl.sum("share_population_value_before"),
              pl.sum("share_population_value_after"),
              pl.sum("share_population_value_change"),
            ]).with_columns([
                pl.col("share_population_value_before").map_elements(formatter).alias("share_population_value_before"),
                pl.col("share_population_value_after").map_elements(formatter).alias("share_population_value_after"),
                pl.col("share_population_value_change").map_elements(formatter).alias("share_population_value_change")
            ]).rename({
                "share_population_value_before": "Total Population Value Before",
                "share_population_value_after": "Total Population Value After",
                "share_population_value_change": "Total Population Value Change"
            })

        with pl.Config(float_precision=0):
            itables.show(merged.with_columns([
                pl.col("share_population_before").map_elements(formatter).alias("share_population_before"),
                pl.col("share_population_after").map_elements(formatter).alias("share_population_after"),
                pl.col("share_population_value_before").map_elements(formatter).alias("share_population_value_before"),
                pl.col("share_population_value_after").map_elements(formatter).alias("share_population_value_after"),
                pl.col("share_population_change").map_elements(formatter).alias("share_population_change"),
                pl.col("share_population_value_change").map_elements(formatter).alias("share_population_value_change")
            ]), paging=False, pageLength=100)
            itables.show(total)

class Simulation:
    current_model: LeaguesModel
    prior_model: LeaguesModel
    league_stats_calculator: pyrust.LeagueStatsCalculator

    def __init__(self, leagues_stats_calculator: pyrust.LeagueStatsCalculator = pyrust.LeagueStatsCalculator()):
        self.current_model = LeaguesModel(leagues_stats_calculator)
        self.league_stats_calculator = leagues_stats_calculator

    def add_league(self, league_name: str):
        self.current_model.load_leagues([league_name])
    
    def render_current_model(self, league_name: str):
        self.current_model.compute_league_shares(league_name)  
        self.current_model.heatmap_counties(league_name)
        return self.current_model.render_map(league_name)  
    
    def apply_changes(self, league_name: str, mutator: Callable[LeaguesModel, None], same_map = False):
        if same_map:
            self.prior_model = self.current_model.copy_with_just_league(league_name)
        else:
            self.prior_model = self.current_model
            self.current_model = self.prior_model.copy_with_just_league(league_name)
        mutator(self.current_model)

    def update_team(self, league_name: str, team_name: str, new_team:Team, same_map = False):
        self.apply_changes(league_name, lambda model: model.update_team(league_name, team_name, new_team), same_map)

    def add_teams(self, league_name: str, new_teams: list[Team], same_map = False):
        self.apply_changes(league_name, lambda model: model.add_teams(league_name, new_teams), same_map)

    def with_league_stats_calculator(self, league_name: str, mutator: Callable[pyrust.LeagueStatsCalculator, None]):
        self.prior_model = self.current_model
        self.current_model = self.prior_model.copy_with_just_league(league_name)
        mutator(self.current_model.league_stats_calculator)

    def show_comparisons(self, league_name: str):
        if self.prior_model:
           self.prior_model.show_pre_post_merged_results(league_name, self.current_model)