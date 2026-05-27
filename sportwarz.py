import json
import sys
import numpy as np
import polars as pl
import ipywidgets as widgets
import itables
import geopandas as gpd

from typing import Callable, TypedDict, Union, Dict, NewType
from IPython.display import display
from ipyleaflet import Map, GeoJSON, Popup, FullScreenControl, basemaps
from pyproj import Transformer
from shapely.ops import transform, unary_union
from shapely.geometry import LineString
from shapely.geometry.point import Point

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

crs5070_transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
crs4326_transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

METERS_TO_MILES = 0.000621371    

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
    _counties_gdf: gpd.GeoDataFrame
    _counties_geojson: CountiesGEOJson
    _leagues: Leagues = { }
    _us_median_income: float

    _geojson_layer: GeoJSON = None
    _leaflet_map: Map = None

    competition_temperature_base : float = 1.25 # Lower -> winner takes it
    non_us_multiplier : float = 2 # part of distance over outside of US including bodies of water.
    not_nearest_multiplier : float = 3 # added to distance multiplied (d - nearest_d) 
    not_same_state_multiplier : float = 2 # distance multiplied if team is not in the same state as the county. 
    canada_multiplier : float = 2 # an additional multiplier if the team is in Canada, as it's even less likely for a county to support a team from another country    
    distance_decay_numerator : float = 0.0025 # multiplied by league weight and divided by team N to get distance decay factor
    use_direct_distance = True

    def load_counties_data(self):
        gdf = gpd.read_file("counties.geojson").set_index("GEOID").to_crs(epsg=5070)

        county_dataframe: pl.DataFrame = pl.read_csv("./co-est2024-alldata.csv", encoding="iso-8859-1")\
            .with_columns((
                pl.col("STATE") * 1000 + pl.col("COUNTY")
        ).cast(pl.Int64).alias("GEOID"))

        co_income_dataframe: pl.DataFrame = pl.read_csv("./Income2023.csv", encoding="iso-8859-1")\
            .with_columns((
                pl.col("FIPS_Code").cast(pl.Int64)
        ).alias("GEOID"))      

        self._us_median_income :float = co_income_dataframe.filter(pl.col("Area_Name") == "United States")["Median_Household_Income"].item()
        gdf = gdf.merge(county_dataframe.to_pandas(), on="GEOID", how="left").merge(co_income_dataframe.to_pandas(), on="GEOID", how="left")
        # only consider the contiguous US for distance calculations, as Alaska and Hawaii are outliers
        gdf = gdf[(gdf["STNAME"] != "Alaska") & (gdf["STNAME"] != "Hawaii")]\
            [["GEOID", "STNAME", "CTYNAME", "geometry", "POPESTIMATE2020", "Median_Household_Income"]]
        gdf["centroid"] = gdf.centroid 
        self._counties_gdf = gdf

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

    def line_lengths(self, p1, p2, the_us):
        # p1, p2 are (lat, lon)
        line = LineString([p1, p2])

        us_part = line.intersection(the_us)
 
        return us_part.length * METERS_TO_MILES, (line.length - us_part.length) * METERS_TO_MILES

    def calculate_league_distances(self, league_name):
        # Dissolve to states
        states = self._counties_gdf.dissolve(by="STNAME", aggfunc="first").reset_index()
        # (Optional) simplify AFTER dissolve
        states["geometry"] = states.simplify(5000, preserve_topology=True) 

        the_us = unary_union(states.geometry).simplify(tolerance=5000, preserve_topology=True)

        teams = self._leagues[league_name]["teams"]

        for team in teams:
            p_wgs84 = Point(team["coordinates"]["lon"], team["coordinates"]["lat"])
            team["coordinates_point"] = transform(crs5070_transformer.transform, p_wgs84)
        
        d = np.zeros((len(self._counties_gdf), len(teams)))

        for i, row in enumerate(self._counties_gdf.itertuples()):
          if not row.centroid.is_empty:
            for j, t in enumerate(teams):
                if self.use_direct_distance:
                    d[i, j] = row.centroid.distance(t["coordinates_point"]) * METERS_TO_MILES         
                else:
                    distances = self.line_lengths(
                        row.centroid,
                        t["coordinates_point"],
                        the_us)
                    d[i, j] = distances[0] + distances[1] * self.non_us_multiplier                                        
   
        self._leagues[league_name]["distances"] = d       

    def calculate_distances(self):
        for league_name in self._leagues.keys():
            self.calculate_league_distances(league_name) 

    def compute_league_shares(self, league):
        league_weight = self._leagues[league]["weight"]
        league_distance_decay = self.distance_decay_numerator / league_weight 

        d = self._leagues[league]["distances"] 

        dataframe_map : dict[str, pl.DataFrame] = { }

        for i, row in enumerate(self._counties_gdf.itertuples()):
            nearest_d = d[i].min()
            nearest_effective_d = 1000000
            county_state = row.STNAME
            R = np.zeros_like(d[i])

            value_multipliers = np.zeros_like(d[i])
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

                #short term enthusiasm dissipates faster             
                DS = np.exp(-team_distance_decay * effective_d * 2)  
                
                R[j] = ((league_weight * 10) + t["L"]  * D)  + ((league_weight * 10) + t["S"] * DS) 

                # very long distances are capped to avoid numerical issues and reflect that beyond a certain point,
                # distance doesn't matter as much (e.g. 3000 miles is not that different from 6000 miles in terms of fan support)
                value_multipliers[j] = np.exp(-min(effective_d, 500) / 100) * league_weight

            expR = np.exp(R / self.competition_temperature_base)
          
            shares = expR / expR.sum(keepdims=True)

            dataframe_out_by_team = {}
            for j, t in enumerate(self._leagues[league]["teams"]):                
                share_population = row.POPESTIMATE2020 * shares[j]
                income_rel = row.Median_Household_Income / self._us_median_income 
                # if the county has higher income than median, it is more valuable to the team, as they can spend more on tickets,
                # merchandise, etc. However, this effect has diminishing returns, as a county with 200k income is not twice as valuable
                # as a county with 100k income. The 0.75 in the denominator controls how quickly the returns diminish.
                income_mult = income_rel / (income_rel + 0.75)
                share_population_value = share_population\
                    * income_mult\
                    * value_multipliers[j]

                if dataframe_out_by_team .get(t["name"]) is None:
                    dataframe_out_by_team[t["name"]] = {
                        "county": row.CTYNAME,
                        "state": row.STNAME,
                        "team_name": t["name"],
                        "share": 0,
                        "share_population": 0,
                        "share_population_value": 0,
                        "color": t.get("color", None)
                    }
                dataframe_out = dataframe_out_by_team[t["name"]]
                # If there are "virtual" teams representing a historical location or minor league presence, 
                # we want to take the max share across all teams representing that location, rather than summing,
                # as they are not additive in terms of fan support.                 
                dataframe_out["share"] = max(dataframe_out["share"], shares[j])
                dataframe_out["share_population"] = max(dataframe_out["share_population"], share_population)
                dataframe_out["share_population_value"] = max(dataframe_out["share_population_value"], share_population_value)
            
            dataframe_out = list(dataframe_out_by_team.values())

            dataframe_map[row.GEOID] = pl.DataFrame(dataframe_out)

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

        with open("./counties.geojson") as f:
            self._counties_geojson =  json.load(f)

        for feature in self._counties_geojson["features"]:
            feature["properties"]["style"] = default_style    

    def heatmap_counties(self, league_name: str, share_threshold = 0.01): 
        self.reset_county_styles()
        for feature in self._counties_geojson["features"]:
            GEOID = feature["properties"]["GEOID"]
            county_rows = self._leagues[league_name]["output_dataframe_map"].get(GEOID)
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
            geoid = feature["properties"]["GEOID"]
            row = self._counties_gdf[self._counties_gdf["GEOID"] == geoid].iloc[0]
            centroid = transform(crs4326_transformer.transform, row.centroid)
            population = row.POPESTIMATE2020

            all_county_rows = pl.DataFrame()    
            leagues_rows = ""  
            try:
                for league in popup_leagues.keys():
                    county_rows = self._leagues[league]["output_dataframe_map"][geoid][["team_name", 'state', "share_population", "share_population_value", "share"]]
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
                            <td>{county_row['share_population']:,.0f}</td> 
                            <td>{county_row['share_population_value']:,.0f}</td> 
                        </tr>'''

                popup = Popup(location=(centroid.y, centroid.x), max_width=500,
                    child=widgets.HTML(f'''
                        <table style="border-collapse: collapse;">
                            <caption>{feature["properties"]["Name"]} - Pop: {population:,.0f}</caption>
                            <tr>
                                <th>League</th>
                                <th>Team</th>
                                <th>Pop. Share</th>
                                <th>Pop. Value</th>
                            </tr>
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

            th,td {
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
        leagues_model.use_direct_distance = self.use_direct_distance
        leagues_model.competition_temperature_base = self.competition_temperature_base
        leagues_model.non_us_multiplier = self.non_us_multiplier
        leagues_model.not_nearest_multiplier = self.not_nearest_multiplier
        leagues_model.not_same_state_multiplier = self.not_same_state_multiplier
        leagues_model.canada_multiplier = self.canada_multiplier
        leagues_model.distance_decay_numerator = self.distance_decay_numerator
        leagues_model._us_median_income = self._us_median_income
        leagues_model._counties_gdf = self._counties_gdf.copy()
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

class Simulation:
    current_model: LeaguesModel
    prior_model: LeaguesModel

    def __init__(self):
        self.current_model = LeaguesModel()
        self.current_model.load_counties_data()

    def add_league(self, league_name: str):
        self.current_model.load_leagues([league_name])
        self.current_model.calculate_league_distances(league_name)
        self.current_model.compute_league_shares(league_name)

    def refresh_current_model(self, league_name: str):
        self.current_model.heatmap_counties(league_name)
        self.current_model.refresh_geojson_layer()
        
    def render_current_model(self, league_name: str):
        self.current_model.heatmap_counties(league_name)
        return self.current_model.render_map(league_name)  
    
    def apply_changes(self, league_name: str, mutator: Callable[LeaguesModel, None], same_map = False):
        if same_map:
            self.prior_model = self.current_model.copy_with_just_league(league_name)
        else:
            self.prior_model = self.current_model
            self.current_model = self.prior_model.copy_with_just_league(league_name)
        mutator(self.current_model)
        self.current_model.calculate_league_distances(league_name)
        self.current_model.compute_league_shares(league_name)

    def update_team(self, league_name: str, team_name: str, new_team:Team, same_map = False):
        self.apply_changes(league_name, lambda model: model.update_team(league_name, team_name, new_team), same_map)

    def add_teams(self, league_name: str, new_teams: list[Team], same_map = False):
        self.apply_changes(league_name, lambda model: model.add_teams(league_name, new_teams), same_map)

    def show_comparisons(self, league_name: str):
        if self.prior_model:
           self.prior_model.show_pre_post_merged_results(league_name, self.current_model)


