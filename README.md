This is an attempt to apply a mathematical model to US counties to predict the level of support within. This is inspired by many similar things on the internet like:

- https://public.tableau.com/app/profile/matt.sorenson/viz/NFLFollowerMap/NFLMap

- https://public.tableau.com/app/profile/matt.sorenson/viz/MLBFollowerMap/MLBMap

- https://public.tableau.com/app/profile/matt.sorenson/viz/NBAFollowerMap/NBAMap

- https://public.tableau.com/app/profile/matt.sorenson/viz/NHLFollowerMap/NHLMap

- https://public.tableau.com/app/profile/matt.sorenson/viz/MLSFanMap/MLSMap

The above all use Twitter followers for the counts and many counties do have a rather small sample size.

#### Examples of MLB and NHL Heat Maps

![MLB Heat Map](./MLBHeatMap.png)

![NHL Heat Map](./NHLHeatMap.png)

#### Things taken into account:

- Popularity of the league / sport (league_weight)

- Long-term establishment of the team. This could called allegiance, equity, etc. In in the code referred as "L"
    
    - Teams gain this by over time:
    
        - Simply existing and accelerated or slowed by having positive/zero or negative short-term enthusiasm.        

- Short-term enthusiasm. In the code referred as "S"

    - Expansion and relocated teams.

    - Team is performing well though highly established teams will not be as enthused as this is expected of them. 
    
    - Team has phenom or marquee player

- National footprint. In the code, "N".

    - Large markets 

    - Dedicated broadcaster like WGN (Chicago Cubs), TBS (Atlanta Braves) and KMOX Radio historically for the St. Louis Cardinals.

    - The marquee or phenom can do this too. Messi, Lebron, Ohtani, etc

- Distance from venue dissipates loyalty. Higher "N" value increases "range".
    
    - Teams other than the nearest team with have an effective distance that is longer. 

    - Teams in other states will have a effective distance multiplier. Even bigger for teams in Canada
    
The code does none of the "L", "S" or "N" accumlation over time. The current values are subjective. Mentally modelling them would include:

- "L"
    
    - Team has existed in the same market for long time. Stagnates or even goes down with prolonged apathy indicated by "S". Example of losing "L" is the Pittsburgh Pirates. 

- "S":
    
    - Maybe reduced by period where other team in market have higher "S". Example would be St. Louis football Cardinals in 1980s when the long established baseball Cardinals played in three World Series'. Inability to build "L" when in theory a football team should be "top dog" (higher league_weight) in the market led to the teams relocation to Phoenix.
    
    - New venue that over time helps build "L"

- "N":

    - Larger markets teams with big name rosters will have higher value here. 

    - Teams with good reputations from the days a given league was smaller     

#### "Fun" I'd like to be able to do

- Model expansion and relocation to see if fans are gained and how existing fan bases might shift

    - Even silly things like major city in Delaware getting teams like the Metropolis Meteors.

- Model what might happen if a league implemented some kind of pro/rel scheme. Enthusiasm for such team would go negative and long term allegiance would to start tp lower if promotion is earned back quickly.


#### Thing I am trying to add.

- Residual loyalty to a relocated team like the Raiders in LA.

- In baseball, having a minor league affiliate in a remote county should increase loyalty even over a closer market.

    - The model might favor Cardinals and Royals over Cubs in Iowa. However, the Cubs have AAA in Des Moines and seem favored in the state.

Experimenting with having a "virtual" team in location with same name as parent club or former club to represent both above

#### Things I'd like to add.

- ~~"America's Teams": The NY Times maps shows Yankees, Red Sox, Lakers, Heat fandom in remote places. Perhaps there should be a bottom to a teams distance degradation.~~ This is accomplished using the "N" value.

- Presences of lower league teams or major college programs in basketball and football could impact pro fandom. This is more of a factor with modeling the "L", "S" values.

- Competition for the other sports

- Oh, Canada!

- Simulate upstart leagues. Instead of leagues have a higher level "sport" category. Upstart leagues would have lower "L" values but could have a bit of "S".

#### Quirks that trickier to mitigate

- Distance is direct "as the crow flies". This leads to quirks where teams in western Michigan are more loyal to Green Bay than they likely are. 

- A farther market may be easier to travel to. Springfield Missouri is closer to Kansas City than St. Louis, but has more Cardinals fans. Recently the Cardinal put a AA affiliate there but mostly because I-44. This is a bit mitigated by the having the AA team in the data.

- Since the NFL's Giants and Jets play in the same venue, it will be a challenge to program in the geographic tendencies on those two teams. This could be mitigate be having a "virtual" teams in areas.

