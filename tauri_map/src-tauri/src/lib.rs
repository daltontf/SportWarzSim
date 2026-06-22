use rust_calc::{LeagueStatsCalculator, League, LeagueStats};
use tauri::{Manager, State};

use std::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(Mutex::new(LeagueStatsCalculator::new_default()));
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![load_league])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command] 
fn load_league(state: State<'_, Mutex<LeagueStatsCalculator>>, league_data: &str) -> LeagueStats {
    let league: League = serde_json::from_str(league_data).unwrap();

    let calculator = state.lock().unwrap();

    calculator.load_league(league)
}