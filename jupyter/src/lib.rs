use pyo3::pymodule;

#[pymodule]
mod pyrust {
    use pyo3::prelude::*;
    use pythonize::{depythonize, pythonize};
    use rust_calc::{LeagueStatsCalculator, League};

    #[pyclass]
    struct PyoLeagueStatsCalculator {
        league_stats_calculator: LeagueStatsCalculator
    }

    #[pymethods]
    impl PyoLeagueStatsCalculator {
        #[new]
        fn new() -> Self {
            let league_stats_calculator = LeagueStatsCalculator::new_default();

            Self {
                // outside_lower48_multiplier: 2.0,
                // not_nearest_multiplier: 2.0,
                // non_same_state_multiplier: 2.0,
                // distance_decay_numerator: 0.0025,
                // competition_temperature_base: 1.00,
                league_stats_calculator
            }
        }

        fn load_league(&self, py: Python<'_>, league_data: &Bound<'_, PyAny>) -> Py<PyAny> {
            let league: League = depythonize(league_data).unwrap();

            let league_stats = self.league_stats_calculator.load_league(&league);

            pythonize(py, &league_stats).unwrap().into()
        }
    

        fn lookup_state_name_by_coordinates(&self, latitude: f64, longitude: f64) -> Option<String> {
            self.league_stats_calculator.lookup_state_name_by_coordinates(latitude, longitude)
        }
    }
}
