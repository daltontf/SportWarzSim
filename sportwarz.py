import json
import sys
import shapely
import numpy as np
import polars as pl
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
    dataframe: pl.DataFrame
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
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
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

def league_teams_sums(league_data) -> pl.DataFrame:
    league_df = pl.concat(league_data["output_dataframe_map"].values())
    return league_df.select(['team_name', 'share_population', 'share_population_value'])\
            .group_by('team_name')\
            .agg(pl.sum('share_population'), pl.sum('share_population_value'))\
            .sort('share_population_value', descending=True)

class LeaguesModel:
    _counties_geojson: CountiesGEOJson
    _counties_centroids: dict[tuple[str, str], shapely.geometry.Point] = { }
    _counties_median_incomes: dict[str, float] = { }
    _leagues: Leagues = { }
    _county_dataframe: pl.DataFrame  
    _us_median_income: float

    _geojson_layer: GeoJSON = None
    _leaflet_map: Map = None

    competition_temperature_base : float = 1 # Lower -> winner takes it
    not_nearest_multiplier : float = 2 # added to distance multiplied (d - nearest_d) 
    not_same_state_multiplier : float = 2
    canada_multiplier : float = 2 

    def load_counties_geojson(self):
        with open("./counties.geojson") as f:
            self._counties_geojson =  json.load(f)
        
    def load_counties_data(self):
        self._county_dataframe = pl.read_csv("./co-est2024-alldata.csv", encoding="iso-8859-1")

        co_income_dataframe: pl.DataFrame = pl.read_csv("./Income2023.csv", encoding="iso-8859-1")

        for row in co_income_dataframe.iter_rows(named=True):
            self._counties_median_incomes[row["FIPS_Code"]] = row["Median_Household_Income_2022"]

        self._us_median_income :float = self._counties_median_incomes[0]

        for feature in self._counties_geojson["features"]:
            coords_stack = [ feature["geometry"]["coordinates"] ]

            while isinstance(coords_stack[-1][0], list):
                coords_stack.append(coords_stack[-1][0]) 

            coords = coords_stack[-2]    
            try: 
                raw_polygon = shapely.geometry.Polygon(coords) 
                state : str = feature["properties"]["STATEFP"]
                county : str = feature["properties"]["COUNTYFP"]
                self._counties_centroids[(state, county)] = raw_polygon.centroid
            except Exception as e:
                print(e)
                print("Invalid coordinates:", feature["properties"]["Name"])  

    def read_league(self, league_name: str):
        with open(f"./teams_{league_name}.json") as f:
            return f.read()

    def load_league(self, league_dict)-> str:
        league_name : str = league_dict["league_name"]
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
        
        counties = self._county_dataframe.to_dicts()

        d = np.zeros((len(counties), len(teams)))

        for i, c in enumerate(counties):
            centroid = self._counties_centroids.get((c["STATE"], c["COUNTY"]))
            if centroid:
                for j, t in enumerate(teams):
                    d[i, j] = haversine_miles(
                        centroid,
                        t["coordinates"]["lat"],
                        t["coordinates"]["lon"]
                    )
   
        self._leagues[league_name]["distances"] = d       

    def calculate_distances(self):
        for league_name in self._leagues.keys():
            self.calculate_league_distances(league_name) 

    def compute_league_shares(self, league):
        league_weight = self._leagues[league]["weight"]
        league_distance_decay = .002 / league_weight 

        d = self._leagues[league]["distances"] 

        dataframe_map : dict[str, pl.DataFrame] = { }

        counties = self._county_dataframe.to_dicts()

        for i, c in enumerate(counties):
            nearest_d = d[i].min()
            nearest_effective_d = 1000000
            county_state = c["STNAME"]
            R = np.zeros_like(d[i])

            distance_value_multipliers = np.zeros_like(d[i])
            for j, t in enumerate(self._leagues[league]["teams"]):
                effective_d = d[i][j]
                if effective_d > nearest_d:
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
                
                N: float = t.get("N", 5.0)
                team_distance_decay : float = league_distance_decay * (5/N) 
                # teams with higher N are less affected by distance, as they have more resources to reach fans further away

                D = np.exp(-team_distance_decay * min(effective_d, 15000/N)) 
                # very long distances are capped to avoid numerical issues and reflect that beyond a certain point,
                #  distance doesn't matter as much (e.g. 3000 miles is not that different from 6000 miles in terms of fan support)
                
                DS = np.exp(-team_distance_decay * effective_d * 2)  #short term enthusiasm dissipates faster             
                
                R[j] = ((league_weight * 10) + t["L"]  * D)  + ((league_weight * 10) + t["S"] * DS) 

                distance_value_multipliers[j] = np.exp(-min(effective_d, 500) / 200) * league_weight

            expR = np.exp(R / self.competition_temperature_base)
          
            shares = expR / expR.sum(keepdims=True)

            dataframe_out_by_team = {}
            for j, t in enumerate(self._leagues[league]["teams"]):                
                share_population = c["POPESTIMATE2020"] * shares[j]
                income_rel = self._counties_median_incomes[c["STATE"] * 1000 + c["COUNTY"]] / self._us_median_income 
                # if the county has higher income than median, it is more valuable to the team, as they can spend more on tickets,
                # merchandise, etc. However, this effect has diminishing returns, as a county with 200k income is not twice as valuable
                # as a county with 100k income. The 0.75 in the denominator controls how quickly the returns diminish.
                income_mult = income_rel / (income_rel + 0.75)
                share_population_value = share_population\
                    * distance_value_multipliers[j]\
                    * income_mult

                if dataframe_out_by_team .get(t["name"]) is None:
                    dataframe_out_by_team[t["name"]] = {
                        "county": c["CTYNAME"],
                        "state": c["STNAME"],
                        "team_name": t["name"],
                        "share": 0,
                        "share_population": 0,
                        "share_population_value": 0,
                        "color": t.get("color", None)
                    }
                dataframe_out = dataframe_out_by_team[t["name"]]
                dataframe_out["share"] += shares[j]
                dataframe_out["share_population"] += share_population
                dataframe_out["share_population_value"] += share_population_value    
            
            dataframe_out = list(dataframe_out_by_team.values())

            dataframe_map[(c["STATE"], c["COUNTY"])] = pl.DataFrame(dataframe_out)

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
                county_row = dict(zip(county_rows.columns, county_rows.sort("share_population_value", descending=True).row(0)))
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
            row = self._county_dataframe.filter((pl.col("STATE") == statefp) & (pl.col("COUNTY") == countyfp))
            centroid = self._counties_centroids[(statefp, countyfp)]
            population = row["POPESTIMATE2020"].item(0)

            all_county_rows = pl.DataFrame()    
            leagues_rows = ""  
            try:
                for league in popup_leagues.keys():
                    county_rows = self._leagues[league]["output_dataframe_map"][(statefp, countyfp)][["team_name", 'state', "share_population_value", "share"]]
                    county_rows = county_rows.group_by(['team_name', 'state']).sum() 
                    county_rows = county_rows.with_columns(pl.lit(league).alias("league"))
                    all_county_rows = pl.concat([all_county_rows, county_rows])
      
                if all_county_rows.height == 0:
                    return
                all_county_rows_dict = all_county_rows.sort("share_population_value", descending=True).to_dicts()
                for i, county_row in enumerate(all_county_rows_dict):
                    if county_row["share"] > 1/len(self._leagues[league]["teams"]):
                        leagues_rows += f'''
                        <tr>  
                            <td>{county_row['league']}</td> 
                            <td>{county_row['team_name']}</td> 
                            <td>{county_row['share_population_value']:,.0f}</td> 
                        </tr>'''

                popup = Popup(location=(centroid.y, centroid.x), 
                    child=widgets.HTML(f'''
                        <table style="border-collapse: collapse;">
                            <caption>{feature["properties"]["Name"]} - {population:,.0f}</caption>
                            {leagues_rows}
                        </table>'''))  
            except Exception as e:
                e_type, e_object, e_traceback = sys.exc_info()
                popup = Popup(location=(centroid.y, centroid.x), 
                    child=widgets.HTML(f"""<pre>{e}</pre><pre>Line: {e_traceback.tb_lineno}</pre>>"""))
            leaflet_map.add(popup)
        return show_teams  

    def render_map(self, only_league: str) -> Map:
        if only_league: # Only show pop-up for one league
            popup_leagues = { only_league: self._leagues[only_league] } 
        else:
            popup_leagues = self._leagues     

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

            td {
                padding: 0 10px;
            }
            </style>"""))

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
        leagues_model._county_dataframe = self._county_dataframe
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

        before_sums = league_teams_sums(self._leagues[league_name])\
            .rename({"share_population": "share_population_before", "share_population_value": "share_population_value_before"})

        after_sums = league_teams_sums(after_model._leagues[league_name])\
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
        