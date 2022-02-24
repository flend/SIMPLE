# Teal and pink experiments Feb 2022

The repository setup with a full implementation of the teal and pink cards from Go Nuts With Donuts. The card distribution was from the vanilla game (so few pink cards).
I also note that I made a mistake with the CF implementation - the non-chosen card is not returned to the deck, but I can't see this affecting the model.

## Training run 1

5 hr training (at that point the trainer crashed with no logs, probably not a GNFD business logic issue).

10 models created [model zoo](app/zoo/gonutsfordonuts/teal_and_pink_20220223)

Tournament, 50 games, modelX vs modelY vs base `docker-compose exec app python3 tournament.py -st 0 -sp 11 -d -g 50 -e gonutsfordonut`

### Results
[results file](app/viz/tournament_results_teal_and_pink_50.csv)

Showing players 1 (rows) and 2 (columns) with player 1 score (player 3, always base is hidden)
<img src="app/viz/tournament_teal_and_pink_50.png" alt="hi" class="inline"/>

Showing players 1 (rows) and 2 (columns) with player 2 score (player 3, always base is hidden)
<img src="app/viz/tournament_teal_and_pink_50_score1.png" alt="hi" class="inline"/>

Note that player 3 (base) always scores badly, so takes a lot of the minus weights. However, we still see a trend to the bottom-left corner (player 1 scores) or top-right corner (player 2 scores)
showing that the models are improving and don't appear to have saturated.

Playing against model 10 it had clearly not trained fully (identical cards had very different weights > 0.1) but it played at a decent level. I wasn't able to judge
it's play on the difficult pink cards.

