# Teal and pink experiments Feb 2022

The repository setup with a full implementation of the teal and pink cards from Go Nuts With Donuts. The card distribution was from the vanilla game (so few pink cards).
I also note that I made a mistake with the CF implementation - the non-chosen card is not returned to the deck, but I can't see this affecting the model.

The model architecture was set up with a full 5-player, all-cards observation space, including sight of all discards, and 70 (pick) and 70 (discard) actions.

ACTIONS = 140 # 2x number of card (ids) - 70 for the 5-player game
FEATURE_SIZE = 635

(for reference, in the previous experiments it was a specialised 3-player game space of:
ACTIONS = 39
FEATURE_SIZE = 64
)

## Training run 1

5 hr training (at that point the trainer crashed with no logs, probably not a GNFD business logic issue).

10 models created [model zoo](./../app/viz/teal_and_pink_20220223)

Tournament, 50 games, modelX vs modelY vs base `docker-compose exec app python3 tournament.py -st 0 -sp 11 -d -g 50 -e gonutsfordonut`

### Results
[results file](./../app/viz/tournament_results_teal_and_pink_50.csv)

Showing players 1 (rows) and 2 (columns) with player 1 score (player 3, always base is hidden)
<img src="./../app/viz/tournament_teal_and_pink_50.png" alt="hi" class="inline"/>

Showing players 1 (rows) and 2 (columns) with player 2 score (player 3, always base is hidden)
<img src="./../app/viz/tournament_teal_and_pink_50_score1.png" alt="hi" class="inline"/>

Note that player 3 (base) always scores badly, so takes a lot of the minus weights. However, we still see a trend to the bottom-left corner (player 1 scores) or top-right corner (player 2 scores)
showing that the models are improving and don't appear to have saturated.

Playing against model 10 it had clearly not trained fully (identical cards had very different weights > 0.1) but it played at a decent level. I wasn't able to judge
it's play on the difficult pink cards.

In terms of absolute performance, the models are getting roughly 0.5 against 2x base models.
In contrast, the previous teal-only models got > 0.8 and I'd expect it to be harder for random play (base) to achieve such good results in the more complex teal-and-pink scenario. Note that the previous teal models achieved this performance after 2 generations (5 min of training) so training speed with this setup is vastly worse.

## Training run 2 - b113d6e

Try training a model using the same action / observation space only with teal cards and see how it does against the teal/pink trained models.

In the last experiment in Dec, a teal-only model trained very fast, but this is using the full 5-player all-cards obs/action space so may? train slower.

This trained a lot slower, after about 12 hrs, it had only produced 13 models.

13 models created [model zoo](./../app/viz/teal_only_20220223)

Running a tournament just with these 13 models against each other to see if they have reached saturation (like the previous teal only trials)

![13 model experiment](./assets/images/teal_only_non_best_action.png)

Unclear why training here has been even less successful.

None of these models seem significantly better than base, so not worth proceeding further.
