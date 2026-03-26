### Blank.ipynb

This is a more interactive notebook for the more technically savvy. 

#### Load League Tab

Selecting an existing major sports league loads a JSON representation of the league and teams that can be edited before generating a heat map. 

If there was a pre-existing model for the given league already rendered a comparison report is generated. One can load a league and then edit some values and load the same league again and see the effects of the changes on the fan support on a per-team basis.

#### Evaluate Python Tab

This allows Python code to be executed. Needs to be simplified, but here is some example code usable now:  

```
league_name = "NHL"

team_new = {
    "name": "Portland Pioneers",
    "L": 3.0,
    "S": 3.0,
    "N": 3.0,
    "venue": "Portland Park",
    "state": "Oregon",
    "color": "green",
    "coordinates": {
        "lat": 45.5200,
        "lon": -122.6886
     }
}

leagues_model.load_leagues([league_name])
leagues_model.calculate_distances() 
leagues_model.compute_shares()  

if leagues_model.delete_teams(league_name, team_new["name"]):
    leagues_model.calculate_distances() 
    leagues_model.compute_shares()  

before_model = leagues_model.copy_with_just_league(league_name)

leagues_model.add_teams(league_name, [team_new])
leagues_model.calculate_distances() 
leagues_model.compute_shares()  
leagues_model.heatmap_counties(league_name)
leagues_model.refresh_geojson_layer()

show_pre_post_merged_results(league_name, before_model, leagues_model)   
```
