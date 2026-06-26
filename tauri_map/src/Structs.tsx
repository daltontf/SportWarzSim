export interface Team {
    name: string;
    L: number;
    S: number;
    N: number;
    color: string;
    state: string | [string];
    coordinates: {
        lat: number;
        lon: number;
    }
}

export interface League {
    league_name: string,
    weight: number,
    teams: Team[],
}

export interface TeamStats {
    team_name: string,
    share: number,
    share_population: number,
    share_population_value: number,
    color: string | undefined
}

export interface CountyStats {
    county_name: string,
    state_name: string,
    population: number,
    team_stats: [TeamStats],
}

export interface LeagueStats {
    league_name: string,
    county_stats_by_geoid: Map<number, CountyStats>,
}

export interface CalculatorInteface {
  getCalculationsForLeague: (league:any) => Promise<LeagueStats>,
  getStateForCoordinates: (lat:number, lon:number) => Promise<string>
}