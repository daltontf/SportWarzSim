use lambda_http::{Error, Request, RequestExt, RequestPayloadExt, Response, Body, http::Method, http::StatusCode, run, service_fn, tracing};

use std::sync::Arc;

use rust_calc::{LeagueStatsCalculator, League};

mod http_handler;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing::init_default_subscriber();

    let state = Arc::new(LeagueStatsCalculator::new_default());

    let handler = service_fn(|request: Request| {
        let state = Arc::clone(&state);

        async move {
            if request.method() == Method::POST {
                let accepts_gzip = request
                    .headers()
                    .get("accept-encoding")
                    .and_then(|value| value.to_str().ok())
                    .map(|s| s.contains("gzip"))
                    .unwrap_or(false);
            
                if !accepts_gzip {
                    let response = Response::builder()
                        .status(StatusCode::NOT_ACCEPTABLE) // 406
                        .body(Body::from("Only gzip encoding is supported"))
                        .map_err(Box::new)?;
                    return Ok(response);
                }
                if let Ok(league) = request.payload::<League>() {
                    http_handler::calculate_stats_handler(state, league, request.query_string_parameters()).await
                } else {
                    let response = Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .header("Content-Type", "text/plain")
                        .body(Body::from("Missing or Malformed JSON request body."))
                        .map_err(Box::new)?;
                    Ok(response)
                }
            } else if request.method() == Method::GET {
                if let (Some(lat), Some(lon)) = (
                    request.query_string_parameters().first("lat"),
                    request.query_string_parameters().first("lon")) {
                    http_handler::lookup_state_name_by_coordinates(
                        state,
                        lat.to_string().parse::<f64>().unwrap(),
                        lon.to_string().parse::<f64>().unwrap()
                    ).await
                } else {
                    let response = Response::builder()
                        .status(StatusCode::BAD_REQUEST) // 400
                        .header("Content-Type", "text/plain")
                        .body(Body::from("Missing 'lon' and/or 'lat' query parameters"))
                        .map_err(Box::new)?;
                    Ok(response)
                }   
            } else {
                let response = Response::builder()
                    .status(StatusCode::METHOD_NOT_ALLOWED) // 405
                    .header("Content-Type", "text/plain")
                    .body(Body::from("Only GET or POST method are allowed"))
                    .map_err(Box::new)?;
                Ok(response)
            }
        }
    });
    
    run(handler).await?;
    Ok(())
}

