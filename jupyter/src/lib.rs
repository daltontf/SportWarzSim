use pyo3::pymodule;

#[pymodule]
mod pyrust {
    use pyo3::prelude::*;
    use pythonize::{depythonize, pythonize};
    use rust_calc::{LeagueStatsCalculator, League};
    use std::collections::HashMap;

    #[pyclass]
    struct PyoLeagueStatsCalculator {
        league_stats_calculator: LeagueStatsCalculator
    }

    #[pymethods]
    impl PyoLeagueStatsCalculator {
        #[new]
        fn new() -> Self {
            Self {
                league_stats_calculator: LeagueStatsCalculator::new_default()
            }
        }

        fn load_league_with_overrides(&self, py: Python<'_>, league_data: &Bound<'_, PyAny>, overrides: HashMap<String, f64>) -> Py<PyAny> {
            let league: League = depythonize(league_data).unwrap();
            
            let league_stats = self.league_stats_calculator.load_league_with_overrides(&league, overrides);

            pythonize(py, &league_stats).unwrap().into()
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
