use pyo3::pymodule;

#[pymodule]
mod pyrust {
    use geo::MapCoordsInPlace;
    use geo_types::{Coord, Geometry, Point, line_string};
    use geojson::GeoJson;
    use geos::Geom;
    use ndarray::Array1;
    use ndarray_stats::QuantileExt;
    use proj::Proj;
    use pyo3::prelude::*;
    use pythonize::{depythonize, pythonize};
    use rayon::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fs;
    use std::fs::File;
    use std::io::BufReader;
    use std::time::Instant;
    use wkt::TryFromWkt;

    pub const EPSG_4326: &str = "EPSG:4326";
    pub const EPSG_5070: &str = "EPSG:5070";
    pub const METERS_TO_MILES: f64 = 0.000621371;

    #[derive(Deserialize)]
    pub struct CountyAll {
        #[serde(alias = "GEOID")]
        pub geoid: u32,
        #[serde(alias = "STNAME")]
        pub state_name: String,
        #[serde(alias = "CTYNAME")]
        pub county_name: String,
        #[serde(alias = "POPESTIMATE2020")]
        pub population: u32,
        #[serde(alias = "MEDIAN_HOUSEHOLD_INCOME")]
        pub median_income: u32,
        #[serde(alias = "CENTROID")]
        pub centroid: String,
    }

    #[derive(Deserialize)]
    struct LatLonCoords {
        lat: f64,
        lon: f64,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StateOrStates {
        State(String),
        States(Vec<String>),
    }

    #[derive(Deserialize)]
    struct Team {
        name: String,
        #[serde(alias = "L")]
        value_l: f64,
        #[serde(alias = "S")]
        value_s: f64,
        #[serde(alias = "N")]
        value_n: f64,
        coordinates: LatLonCoords,
        state: StateOrStates,
        color: Option<String>, // Color = None indicates the team is not part of the league
    }

    #[derive(Deserialize)]
    struct League {
        league_name: String,
        weight: f64,
        teams: Vec<Team>,
    }

    #[derive(Serialize,Deserialize)]
    struct State4326 {
        stname: String,
        geometry: Geometry,
    }

    #[derive(Serialize)]
        struct TeamStats {
        team_name: String,
        share: f64,
        share_population: f64,
        share_population_value: f64,
        color: Option<String>,
    }

    #[derive(Serialize)]
    struct CountyStats {
        county_name: String,
        state_name: String,
        population: u32,
        team_stats: Vec<TeamStats>,
    }

    #[derive(Serialize)]
    struct LeagueStats {
        league_name: String,
        county_stats_by_geoid: HashMap<u32, CountyStats>,
    }

    #[pyclass]
    struct LeagueStatsCalculator {
        #[pyo3(get, set)]
        outside_lower48_multiplier: f64,
        #[pyo3(get, set)]
        not_nearest_multiplier: f64,
        #[pyo3(get, set)]
        non_same_state_multiplier: f64,
        #[pyo3(get, set)]
        distance_decay_numerator: f64,
        #[pyo3(get, set)]
        competition_temperature_base: f64,

        all_data: Vec<CountyAll>,
        us_median_income: f64,
        geos_us: geos::Geometry,
        state_geometries: Vec<State4326>
    }

    #[pymethods]
    impl LeagueStatsCalculator {
        #[new]
        fn new() -> Self {
            let file = File::open("counties_all.csv").unwrap();
            let mut rdr = csv::Reader::from_reader(file);

            // Collect the iterator directly into a Result<Vec<Record>>
            let all_data: Vec<CountyAll> = rdr.deserialize().collect::<Result<_, _>>().unwrap();

            let us_median_income =
                all_data.iter().map(|c| c.median_income).sum::<u32>() as f64 / all_data.len() as f64;

            let us_geometry_text = fs::read_to_string("us_geometry.geojson").unwrap();
            let us_geometry_geojson: GeoJson = us_geometry_text.parse().unwrap();
            let us_geometry: Geometry = us_geometry_geojson.try_into().unwrap();
            let geos_us: geos::Geometry = (&us_geometry).try_into().unwrap();

            // let file_reader = BufReader::new(File::open("states-4326.geojson").unwrap());
            // Create a Vec of Country structs from the GeoJSON
            // let state_geometries: Vec<State4326> = geojson::de::deserialize_feature_collection_to_vec::<State4326>(file_reader)
            //     .unwrap()
            //     .into_iter()
            //     .collect();
            let states_reader = BufReader::new(File::open("states-4326.geojson").unwrap());    
            let features: geojson::FeatureCollection =  serde_json::from_reader(states_reader).unwrap();
            let state_geometries: Vec<State4326> = features.features.into_iter().map(|feature| {
                let stname = feature.properties.as_ref().unwrap().get("stname").unwrap().as_str().unwrap().to_string();
                let geometry = feature.geometry.unwrap().value.try_into().unwrap();
                State4326 { stname, geometry }
            }).collect();

            Self {
                outside_lower48_multiplier: 2.0,
                not_nearest_multiplier: 2.0,
                non_same_state_multiplier: 2.0,
                distance_decay_numerator: 0.0025,
                competition_temperature_base: 1.00,
                all_data,
                us_median_income,
                geos_us,
                state_geometries
            }
        }

        fn load_league(&self, py: Python<'_>, league_data: &Bound<'_, PyAny>) -> Py<PyAny> {
            let league: League = depythonize(league_data).unwrap();

            let transform = Proj::new_known_crs(EPSG_4326, EPSG_5070, None).unwrap();

            let teams_coords5070: Vec<Point> = league
                .teams
                .iter()
                .map(|team| {
                    let mut team_coords = Point::new(team.coordinates.lon, team.coordinates.lat); // lon/lat (EPSG:4326)
                    team_coords.map_coords_in_place(|coord: Coord<f64>| {
                        transform.project(coord, false).unwrap()
                    });
                    team_coords
                })
                .collect();

            let league_distance_decay = self.distance_decay_numerator / league.weight;

            let start = Instant::now();

            let all_county_map: HashMap<u32, CountyStats> = self.all_data
                .par_iter()
                .map_init(|| Geom::clone(&self.geos_us).unwrap(), // geos_us is not thread-safe
                    |geos_us_clone, county| {
                        let team_distances: Array1<f64> = teams_coords5070.iter().map(|team_coords| {
                            let centroid: Point<f64> = geo_types::Geometry::try_from_wkt_str(county.centroid.as_str()).unwrap().try_into().unwrap(); // already in EPSG:5070
                            let line_string = line_string![(x: team_coords.x(), y: team_coords.y()), ( x: centroid.x(), y: centroid.y())];

                            let line_geos: geos::Geometry = line_string.try_into().unwrap();

                            let intersection = line_geos.intersection(geos_us_clone).unwrap();
                            let line_length = line_geos.length().unwrap() as f64 * METERS_TO_MILES;
                            let intersection_length = intersection.length().unwrap() as f64 * METERS_TO_MILES;

                            intersection_length + ((line_length - intersection_length) * self.outside_lower48_multiplier)
                        }).collect();

            let nearest_distance = *team_distances.min().unwrap();
            let mut nearest_effectivly = 1000000.0;
            let mut value_r:Array1<f64> = Array1::zeros(team_distances.len());
            let mut value_multipliers:Array1<f64> = Array1::zeros(team_distances.len());

            for (i, team) in league.teams.iter().enumerate() {
                let mut effective_distance = team_distances[i];
                if effective_distance > nearest_distance {
                    // Every team that is not the nearest has the extra distance multiplied by NOT_NEAREST_MULTIPLER
                    // to increase its effectiv distance relative to the nearest team
                    effective_distance = nearest_distance + ((effective_distance - nearest_distance) * self.not_nearest_multiplier)
                }
                // Any team not in the same state or states if team market spans multiple states (KC)
                // has its distance multiplied by NON_SAME_STATE_MULTIPLIER
                match &team.state {
                    StateOrStates::State(state) if state != &county.state_name => {
                        effective_distance *= self.non_same_state_multiplier;
                    },
                    StateOrStates::States(states) if !states.contains(&county.state_name) => {
                        effective_distance *= self.non_same_state_multiplier;
                    },
                    _ => {}
                }
                if effective_distance < nearest_effectivly {
                    nearest_effectivly = effective_distance;
                }
                let team_distance_decay: f64 = league_distance_decay * (5.0/team.value_n);

                let d_exp = f64::exp(-team_distance_decay * effective_distance);
                
                let ds_exp = f64::exp(-team_distance_decay * effective_distance * 2.0);

                value_r[i] = ((league.weight * 10.0) + team.value_l * d_exp)  + ((league.weight * 10.0) + team.value_s * ds_exp);
                
                value_multipliers[i] = f64::exp(-f64::min(effective_distance, 200.0) / 50.0) * league.weight;
            }
            let exp_r = (value_r / self.competition_temperature_base).mapv(f64::exp);

            let shares = &exp_r / exp_r.sum();

            let mut team_map: HashMap<String, TeamStats> = std::collections::HashMap::new();

            for (i, team) in league.teams.iter().enumerate() {
                let share_population = county.population as f64 * shares[i];
                let income_relative = county.median_income as f64 / self.us_median_income;
                // if the county has higher income than median, it is more valuable to the team, as they can spend more on tickets,
                // merchandise, etc. However, this effect has diminishing returns, as a county with 200k income is not twice as valuable
                // as a county with 100k income. The 0.75 in the denominator controls how quickly the returns diminish.
                let income_multiple = income_relative / (income_relative + 0.75);
                let share_population_value = share_population * income_multiple * value_multipliers[i];

                let team_stats = TeamStats {
                    team_name: team.name.clone(),
                    share: shares[i],
                    share_population,
                    share_population_value,
                    color: team.color.clone()
                };

                // If there are "virtual" teams representing a historical location or minor league presence,
                // we want to take the max share across all teams representing that location, rather than summing,
                // as they are not additive in terms of fan support.
                team_map.entry(team.name.clone())
                    .and_modify(|existing| {
                        existing.share = existing.share.max(team_stats.share);
                        existing.share_population = existing.share_population.max(team_stats.share_population);
                        existing.share_population_value = existing.share_population_value.max(team_stats.share_population_value);
                        if existing.color.is_none() && team.color.is_some() {
                            existing.color = team.color.clone();
                        }
                    })
                    .or_insert(team_stats);
            }

            let mut team_stats:Vec<TeamStats> = team_map.into_values().collect();

            team_stats.sort_by(|a, b| b.share_population_value.partial_cmp(&a.share_population_value).unwrap());

            (county.geoid, CountyStats {
                county_name: county.county_name.clone(),
                state_name: county.state_name.clone(),
                population: county.population,
                team_stats
            })
        }).collect();

            let elapsed = start.elapsed();
            println!("Elapsed: {:?}", elapsed);

            let league_stats = LeagueStats {
                league_name: league.league_name,
                county_stats_by_geoid: all_county_map,
            };

            pythonize(py, &league_stats).unwrap().into()
        }
    

        fn lookup_state_name_by_coordinates(&self, latitude: f64, longitude: f64) -> Option<String> {
            let point = Point::new(longitude, latitude); // Note: GeoJSON uses (lon, lat) order

            for state in &self.state_geometries {
                if let Ok(geos_geometry) = geos::Geometry::try_from(&state.geometry) {
                    if geos_geometry.contains(&geos::Geometry::try_from(&point).unwrap()).unwrap() {
                        return Some(state.stname.clone());
                    }
                }
            }
            None
        }
    }
}
