import { useState } from 'react';
import { AllCommunityModule, themeQuartz, colorSchemeDark } from 'ag-grid-community';
import _ from 'lodash';

import { AgGridProvider, AgGridReact } from 'ag-grid-react';

import { type LeagueStats } from './Structs';

const modules = [AllCommunityModule];

function flattenTeamStats(league_stats: LeagueStats, prefix: string): any {
  const flattened = Object.values(league_stats.county_stats_by_geoid).flatMap(county => county.team_stats); 
  const grouped = _.groupBy(flattened, team => team.team_name);
  
  return Object.entries(grouped).map(([team_name, rows]) => {
    const result = { 
      team_name
    }
    result[`${prefix}_share_population`] = _.sumBy(rows, "share_population");
    result[`${prefix}_share_population_value`] = _.sumBy(rows, "share_population_value");
    return result;
  });
}

export default function CalculationsComparision({ calculations }: { calculations: [LeagueStats | null, LeagueStats] | null }) {
  if (!calculations ||!calculations[1]) return <></>;

  let data = flattenTeamStats(calculations[1], "current");

  const valueFormatter = (column) => column.value?.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  
  let columns = [
      { field: "team_name" },
      { field: "current_share_population", valueFormatter },
      { field: "current_share_population_value", valueFormatter }
  ]

  let summary = null;

  if (calculations[0]) {
    
    const priorData = flattenTeamStats(calculations[0], "prior");

    const priorByKey = _.keyBy(priorData, "team_name");
    const currentByKey = _.keyBy(data, "team_name");

    const keys = _.union(
      Object.keys(priorByKey),
      Object.keys(currentByKey)
    );
    
    data = keys.map(key => ({
      ...priorByKey[key],
      ...currentByKey[key],
    }));

    for (let row of data) {
      row.share_population_change = (row.current_share_population ?? 0) - (row.prior_share_population ?? 0);
      row.share_population_value_change = (row.current_share_population_value ?? 0) - (row.prior_share_population_value ?? 0);
    }

    summary = [{
      total_prior_share_population_value:  _.sumBy(data, "prior_share_population_value"),
      total_current_share_population_value:  _.sumBy(data, "current_share_population_value"),
    }];
    summary[0]["total_population_value_change"] = summary[0]["total_current_share_population_value"] - summary[0]["total_prior_share_population_value"];

    columns = columns.concat([
      { field: "prior_share_population", valueFormatter },
      { field: "prior_share_population_value", valueFormatter },
      { field: "share_population_change", valueFormatter },
      { field: "share_population_value_change", valueFormatter }
    ]);
  }

  const [rowData, setRowData] = useState(data);
  const [colDefs, setColDefs] = useState(columns);

  return (
    <AgGridProvider modules={modules}>
      <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "92vh" }}>
        <div style={{ flex: 1 }}>
        <AgGridReact theme={themeQuartz.withPart(colorSchemeDark)}
            rowData={rowData}
          columnDefs={colDefs}
          />
        </div>
        {summary && <div style={{ minHeight: "90px" }}><AgGridReact theme={themeQuartz.withPart(colorSchemeDark)}
          rowData={summary}
          columnDefs={[
              { field: "total_prior_share_population_value", valueFormatter },
              { field: "total_current_share_population_value", valueFormatter },
              { field: "total_population_value_change", valueFormatter }
            ]}
        /></div>}
      </div>
    </AgGridProvider>
  );
}