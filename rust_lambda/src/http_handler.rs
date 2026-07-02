use lambda_http::{
    Body, Error, aws_lambda_events::query_map::QueryMap, http::{Response, StatusCode}};

use std::sync::Arc;
use std::io::Write;
use std::collections::HashMap;

use flate2::write::GzEncoder;
use flate2::Compression;

use rust_calc::{ League, LeagueStatsCalculator, 
    OUTSIDE_LOWER48_MULTIPLIER_KEY,
    NOT_NEAREST_MULTIPLIER_KEY,
    COMPETITION_TEMPERATURE_BASE_KEY,
    NON_SAME_STATE_MULTIPLIER_KEY,
    DISTANCE_DECAY_NUMERATOR_KEY
};

pub async fn calculate_stats_handler(state: Arc<LeagueStatsCalculator>, payload: Option<League>, query_map: QueryMap) -> Result<Response<Body>, Error> {
    let league = match payload {
        Some(data) => data, // Successfully parsed
        None => {
            // The request was successful, but the body was entirely empty
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "text/plain")
                .body(Body::from("Missing or Malformed JSON request body."))
                .map_err(Box::new)?)
        }
    };

    let mut overrides: HashMap<String, f64> = HashMap::new();

    for name in vec![
        OUTSIDE_LOWER48_MULTIPLIER_KEY,
        NOT_NEAREST_MULTIPLIER_KEY,
        NON_SAME_STATE_MULTIPLIER_KEY,
        DISTANCE_DECAY_NUMERATOR_KEY,
        COMPETITION_TEMPERATURE_BASE_KEY
    ] {
        query_map.first(name)
            .and_then(|s| s.parse::<f64>().ok())
            .inspect(|it| { overrides.insert(name.to_string(), *it); });
    }   

    let result = state.load_league_with_overrides(&league, overrides);

    let body = serde_json::to_string(&result)?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(body.as_bytes())?;
    let compressed_bytes: Vec<u8> = encoder.finish()?;

    println!(
        "returning {} chars compressed to {} bytes",
        body.len(),
        compressed_bytes.len()
    );
   
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .header("Content-Encoding", "gzip")
        .header("Access-Control-Allow-Origin", "*")
        .body(Body::from(compressed_bytes))
        .map_err(Box::new)?)
}

pub async fn lookup_state_name_by_coordinates(state: Arc<LeagueStatsCalculator>, latitude: f64, longitude: f64) -> Result<Response<Body>, Error>{
    if let Some(stname) = state.lookup_state_name_by_coordinates(latitude, longitude) {
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/plain")
            .header("Access-Control-Allow-Origin", "*")
            .body(Body::from(stname.clone()))
            .map_err(Box::new)?)
    } else {
        Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "text/plain")
            .body(Body::from("No state found containing the provided coordinates"))
            .map_err(Box::new)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::Read;

    fn decompressed_bytes(compressed_bytes: &[u8]) -> String {
        let mut decoder = GzDecoder::new(compressed_bytes);
        let mut decompressed_bytes = Vec::new();

        // 4. Read (decompress) the data into the buffer
        decoder.read_to_end(&mut decompressed_bytes).unwrap();

        String::from_utf8(decompressed_bytes).unwrap()
    }

    // #[tokio::test]
    // async fn test_http_handler_happy_path() {
    //     let request = lambda_http::http::Request::builder()
    //         .method("POST")
    //         .header("Content-Type", "application/json")
    //         .body(Body::from(
    //             serde_json::to_string(
    //                 &(League {
    //                     league_name: "MLB".into(),
    //                     weight: 8.0,
    //                     teams: vec![
    //                         Team {
    //                             name: "Chicago Cubs".into(),
    //                             value_l: 9.0,
    //                             value_s: 1.0,
    //                             value_n: 8.0,
    //                             coordinates: LatLonCoords {
    //                                 lat: 41.947201,
    //                                 lon: -87.656413,
    //                             },
    //                             state: StateOrStates::State("Illinois".into()),
    //                             color: Some("blue".into()),
    //                         },
    //                         Team {
    //                             name: "St. Louis Cardinals".into(),
    //                             value_l: 9.0,
    //                             value_s: 1.0,
    //                             value_n: 7.0,
    //                             coordinates: LatLonCoords {
    //                                 lat: 38.626004,
    //                                 lon: -90.188538,
    //                             },
    //                             state: StateOrStates::State("Missouri".into()),
    //                             color: Some("red".into()),
    //                         },
    //                     ],
    //                 }),
    //             )
    //             .unwrap(),
    //         ))
    //         .unwrap();

    //     let response = function_handler(request).await.unwrap();

    //     let body_bytes = response.body().to_vec();

    //     assert_eq!(response.status(), 200);

    //     let decompressed_body_string = decompressed_bytes(&body_bytes);
    //     println!("decompressed_body_string: {}", decompressed_body_string);

    //     let league_stats_json: LeagueStats =
    //         serde_json::from_str(&decompressed_body_string).unwrap();

    //     assert_eq!(league_stats_json.league_name, "MLB");
    //     assert_eq!(league_stats_json.county_stats_by_geoid.len(), 3108);

    //     let stl_county_stats = league_stats_json.county_stats_by_geoid.get(&29189).unwrap();

    //     assert_eq!(stl_county_stats.county_name, "St. Louis County");
    //     assert_eq!(stl_county_stats.state_name, "Missouri");
    //     assert_eq!(stl_county_stats.population, 1003376);
    //     assert_eq!(stl_county_stats.team_stats.len(), 2);

    //     assert_eq!(
    //         stl_county_stats.team_stats[0].team_name,
    //         "St. Louis Cardinals"
    //     );
    //     assert_eq!(stl_county_stats.team_stats[0].color, Some("red".into()));
    //     assert_eq!(stl_county_stats.team_stats[1].team_name, "Chicago Cubs");
    //     assert_eq!(stl_county_stats.team_stats[1].color, Some("blue".into()));
    // }
}