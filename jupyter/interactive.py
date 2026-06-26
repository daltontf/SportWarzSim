import polars as pl
import itables
import ipywidgets as widgets
import json
import ipywidgets as widgets
import os
import pyrust

from ipyleaflet import Map, Popup, Marker
from IPython.display import display
from ipyleaflet import Map, GeoJSON, FullScreenControl, WidgetControl, basemaps

class Interactive:
    league_name: str
    out: widgets.Output
    counties_geojson: None
    league: None
    leagues_data: None
    calculator: pyrust.PyoLeagueStatsCalculator


    def __init__(self, league_name: str):
        self.league_name = league_name
        self.out = widgets.Output(layout = widgets.Layout(
            height = "100%"  
        ))

        os.environ["RUST_BACKTRACE"] = "1"

        with open(f'teams_{league_name}.json', "r") as f:
            self.league = json.load(f)

        with open("./counties_4326.geojson") as f:
            self.counties_geojson =  json.load(f)   

        self.leagues_data = { 

        }

        self.calculator = pyrust.PyoLeagueStatsCalculator() 


    def opacity_for_population(self, share_population_value): 
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

    def reset_county_styles(self):
        default_style = {
            "color": "grey",
            "weight": 1,
            "fillColor": "grey",
            "fillOpacity": 0.0,
        }

        for feature in self.counties_geojson["features"]:
            feature["properties"]["style"] = default_style

    def heatmap_counties(self, league_county_map: dict, share_threshold = 0.01): 
        self.reset_county_styles()
        for feature in self.counties_geojson["features"]:
            geoid = feature["properties"]["geoid"]
            county_data = league_county_map[geoid]
            county_top_team_data = county_data["team_stats"][0]
            if county_top_team_data["share"] > share_threshold:   
                feature["properties"]["style"] = {
                    "color": "grey",
                    "weight": 1,
                    "fillColor": county_top_team_data["color"] ,
                    "fillOpacity": self.opacity_for_population(county_top_team_data["share_population_value"])
                }               
    
    def league_teams_sums(self, county_data) -> pl.DataFrame:
        all_teams = pl.DataFrame([x for county in county_data.values() for x in county["team_stats"]])

        return all_teams.select(['team_name', 'share_population', 'share_population_value'])\
            .group_by('team_name')\
            .agg(pl.sum('share_population'), pl.sum('share_population_value'))\
            .sort('share_population_value', descending=True)  


    def show_comparisons(self, league_calculations_prior, league_calculations_after):
        before_sums = self.league_teams_sums(league_calculations_prior)\
            .rename({"share_population": "share_population_before", "share_population_value": "share_population_value_before"})

        after_sums = self.league_teams_sums(league_calculations_after)\
            .rename({"share_population": "share_population_after", "share_population_value": "share_population_value_after"})

        merged = before_sums.join(after_sums, on='team_name', how='full', coalesce=True)\
            .fill_null(0).with_columns([
                (pl.col("share_population_after") - pl.col("share_population_before")).alias("share_population_change"),
                (pl.col("share_population_value_after") - pl.col("share_population_value_before")).alias("share_population_value_change")
            ])
    
        pl.Config.set_tbl_rows(50)

        formatter = lambda x: f"{x:,.0f}"
    
        with self.out:
            self.out.clear_output(wait=True)
            itables.show(merged.with_columns([
                pl.col("share_population_before").map_elements(formatter),
                pl.col("share_population_after").map_elements(formatter),
                pl.col("share_population_value_before").map_elements(formatter),
                pl.col("share_population_value_after").map_elements(formatter),
                pl.col("share_population_change").map_elements(formatter),
                pl.col("share_population_value_change").map_elements(formatter)
            ]).rename({
               "team_name": "Team",
               "share_population_before": "Share Population Before",
               "share_population_after": "Share Population After",
               "share_population_change": "Share Population Change",
               "share_population_value_before": "Share Population Value Before",
               "share_population_value_after": "Share Population Value After",
               "share_population_value_change": "Share Population Value Change"
            }), paging=False, pageLength=100)

            itables.show(merged.select([
                pl.sum("share_population_value_before"),
                pl.sum("share_population_value_after"),
                pl.sum("share_population_value_change"),
            ]).with_columns([
                pl.col("share_population_value_before").map_elements(formatter),
                pl.col("share_population_value_after").map_elements(formatter),
                pl.col("share_population_value_change").map_elements(formatter)
            ]).rename({
                "share_population_value_before": "Total Population Value Before",
                "share_population_value_after": "Total Population Value After",
                "share_population_value_change": "Total Population Value Change"
            }))

    def calculate_results(self):
        endpoint = os.getenv("REST_CALCULATOR_ENDPOINT")
        if endpoint: # USE REST CALCULATOR
            import requests
            # need to work around marker hack
            clean_teams = {
                **self.league,
                "teams": [
                    {k: v for k, v in team.items() if k != "marker"}
                    for team in self.league["teams"]
                ]
            }
            response = requests.post(endpoint, json= clean_teams)
            result = response.json()
            result["county_stats_by_geoid"] = {int(geoid): value for geoid, value in result["county_stats_by_geoid"].items()}
        else: # USE RUST CALCULATOR
            result = self.calculator.load_league(self.league)
        return result    

    def render_map(self):
        def create_show_teams(leaflet_map:Map, league_county_map: dict):
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

                nonlocal current_popup                
                if current_popup is not None:
                    map.remove(current_popup)        

                current_popup = Popup(location = coordinates, max_width=500,
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
            
                leaflet_map.add(current_popup)
            return show_teams   

        def save_prior_results(event):
            nonlocal prior_league_calculations, result
            prior_league_calculations = result["county_stats_by_geoid"]

        def recalculate_stats(event):
            nonlocal geojson_layer
            try:
                calc_button.description = "Calculating..."
                calc_button.disabled = True
                map.interaction = False

                map.remove_layer(geojson_layer)
 
                result = self.calculate_results()

                self.heatmap_counties(result["county_stats_by_geoid"])
       
                geojson_layer = GeoJSON(data = self.counties_geojson,  hover_style = {"color": "white"})
  
                geojson_layer.on_click(create_show_teams(map, result["county_stats_by_geoid"]))
                map.add_layer(geojson_layer)

                self.show_comparisons(prior_league_calculations, result["county_stats_by_geoid"])
            finally:         
                calc_button.description = "Re-Calculate"
                calc_button.disabled = False
                map.interaction = True

        def create_move_handler(team):
            @self.out.capture()
            def move_handler(event, **kwargs):
                team["coordinates"]["lat"] = kwargs["location"][0]
                team["coordinates"]["lon"] = kwargs["location"][1]
                team["state"] = self.calculator.lookup_state_name_by_coordinates(team["coordinates"]["lat"], team["coordinates"]["lon"])
            return move_handler

        current_popup = None    

        def create_popup_for_team(team, new_team=False):
            team_text = widgets.Text(value=team["name"])
            venue_text = widgets.Text(value=team["venue"])
            l_text = widgets.FloatText(value=team["L"])
            s_text = widgets.FloatText(value=team["S"])
            n_text = widgets.FloatText(value=team["N"])
            if type(team["state"]) is list:
                state_text = widgets.Text(value=", ".join(team["state"]))
            else:
                state_text = widgets.Text(value=team["state"])
            color_picker = widgets.ColorPicker(value=team["color"])
            update_button = widgets.Button(description="Update" if not new_team else "Create", layout=widgets.Layout(width="95%"))
            delete_button = widgets.Button(description="Delete")

            gridbox = widgets.GridBox(
                children=[
                    widgets.Label("Team:"), team_text,
                    widgets.Label("Venue:"), venue_text,
                    widgets.Label("Coordinates:"), widgets.Label(f"({team['coordinates']['lat']}, {team['coordinates']['lon']})"),
                    widgets.Label("L:"), l_text,
                    widgets.Label("S:"), s_text,
                    widgets.Label("N:"), n_text,
                    widgets.Label("State:"), state_text,
                    widgets.Label("Color:"), color_picker,
                    update_button, delete_button if not new_team else widgets.Label("")
                ],
                layout=widgets.Layout(grid_template_columns="100px 1fr"))
    
            def on_update_click(x):
                nonlocal current_popup
                if len(team_text.value.strip()) > 0:
                    team["name"] = team_text.value
                if len(venue_text.value.strip()) > 0:    
                    team["venue"] = venue_text.value
                if l_text.value > 0:
                    team["L"] = l_text.value
                if s_text.value > 0:
                    team["S"] = s_text.value
                if n_text.value > 0:
                    team["N"] = n_text.value
                if len(state_text.value.strip()) > 0:
                    team["state"] = [s.strip() for s in state_text.value.split(",")] if "," in state_text.value else state_text.value.strip()
                if len(color_picker.value.strip()) > 0:
                    team["color"] = color_picker.value 
                if "marker" in team:
                    map.remove_layer(team["marker"])      
                if new_team:
                    self.league["teams"].append(team)
                map.add(create_marker_for_team(team))
                map.remove(current_popup) 
                current_popup = None

            def on_delete_click(x):
                nonlocal current_popup
                # Remove team from league teams list including minor league / virtual teams with same name
                self.league["teams"] = [t for t in self.league["teams"] if t != team and t["name"] != team["name"]]
                map.remove(current_popup) 
                current_popup = None
                map.remove_layer(team["marker"])
       
            update_button.on_click(on_update_click)    
            delete_button.on_click(on_delete_click)
            return gridbox

        def create_marker_click_handler(team):
            @self.out.capture()
            def on_marker_click(**kwargs):
                nonlocal current_popup

                if current_popup is not None:
                    map.remove(current_popup)

                current_popup = Popup(location=kwargs.get("coordinates"), min_width=420, max_width=420)
                current_popup.child = create_popup_for_team(team)
                map.add(current_popup)
            return on_marker_click    

        def create_marker_for_team(team):
            marker = Marker(title = team["name"], location=(team["coordinates"]["lat"], team["coordinates"]["lon"]))
            marker.on_move(create_move_handler(team))
            marker.on_click(create_marker_click_handler(team))
            team["marker"] = marker # Hacky way to keep track of marker for deletion 
            return marker

        @self.out.capture()
        def handle_interaction(**kwargs):
            nonlocal current_popup
            if kwargs.get("type") != "contextmenu":
                return  

            if current_popup is not None:
                map.remove(current_popup)  

            content = create_popup_for_team({
                "name": "New Team",
                "venue": "Stadium Arena",
                "coordinates": {"lat": kwargs.get("coordinates")[0], "lon": kwargs.get("coordinates")[1]},
                "L": 1.0,
                "S": 1.0,
                "N": 1.0,
                "state": self.calculator.lookup_state_name_by_coordinates(kwargs.get("coordinates")[0], kwargs.get("coordinates")[1]),
                "color": "#000000"},
                new_team=True)
            current_popup = Popup(location=kwargs.get("coordinates"), min_width=420, max_width=420)
            current_popup.child = content
            map.add(current_popup)
        
        result = self.calculate_results()
        self.leagues_data[result["league_name"]] = result["county_stats_by_geoid"]

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

        self.heatmap_counties(result["county_stats_by_geoid"])

        map = Map(
            basemap=basemaps.CartoDB.Positron,
            center=[38.72728229549864, -96.9010842308538],
            zoom=5,
            layout=widgets.Layout(width="100%", height="600px"),
            scroll_wheel_zoom=True)

        geojson_layer = GeoJSON(data = self.counties_geojson,  hover_style = {"color": "white"})

        prior_league_calculations = result["county_stats_by_geoid"]

        calc_button=widgets.Button(description='Re-Calculate')
        save_prior=widgets.Button(description='Save Prior Results')


        for team in self.league["teams"]:
            if "color" in team:
                map.add(create_marker_for_team(team))

        self.heatmap_counties(result["county_stats_by_geoid"])
        geojson_layer.data = self.counties_geojson 

        geojson_layer.on_click(create_show_teams(map, result["county_stats_by_geoid"]))

        map.add(geojson_layer)
        map.add(FullScreenControl())
        map.fullscreen = True

        map.on_interaction(handle_interaction)

        calc_button.on_click(recalculate_stats)
        save_prior.on_click(save_prior_results)

        box = widgets.VBox([
            calc_button,
            save_prior
        ])
        
        control = WidgetControl(widget=box, position='bottomleft')
        map.add(control)

        return map
