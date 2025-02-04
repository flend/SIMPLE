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

## Training run 3 - 236b9e5

To see if we can restore fast training, using the teal only cards and hacking the obs/action spaces down to:

ACTIONS = 70
FEATURE_SIZE = 353

by dropping the RV action, reducing to 3 players and dropping the full visibility of the discard deck.

This trained 34 models after in excess of 6 hrs, but performance seemed no better than noise.

![smaller feature set results](./assets/images/teal_only_smaller_model.png)

I'm not quite sure why this performed SO badly, but it encourages me to make the model just as tiny as possible.

## Training run 4 - 079da56

Reverted to the original teal-only codebase to check I could replicate the results and it wasn't a fluke.

![teal only replay feature set results](./assets/images/tournament_results_079da56.png)

The model performed far better (again) than any recent experiment.

Trainin

```
-rw-r--r-- 1 flend flend 140927 Mar  1 07:08 _model_00001_0_0.59_36864_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 07:11 _model_00002_0_0.522_77824_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 07:19 _model_00003_0_0.265_159744_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 07:35 _model_00004_0_0.236_282624_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 07:46 _model_00005_0_0.29_364544_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 07:50 _model_00006_0_0.248_405504_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 08:09 _model_00007_0_0.202_569344_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 08:14 _model_00008_0_0.215_610304_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 14:12 _model_00009_0_0.221_3600384_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 15:28 _model_00010_0_0.269_4214784_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 15:33 _model_00011_0_0.261_4255744_.zip
-rw-r--r-- 1 flend flend 140927 Mar  1 16:47 _model_00012_0_0.209_4911104_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 07:56 _model_00013_0_0.212_16297984_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 08:08 _model_00014_0_0.236_16502784_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 08:17 _model_00015_0_0.212_16666624_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 08:31 _model_00016_0_0.201_16912384_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 08:55 _model_00017_0_0.246_17321984_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 09:23 _model_00018_0_0.222_17813504_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 09:27 _model_00019_0_0.232_17895424_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 09:39 _model_00020_0_0.218_18100224_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 09:53 _model_00021_0_0.203_18345984_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 10:00 _model_00022_0_0.211_18468864_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 11:14 _model_00023_0_0.206_19779584_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 12:34 _model_00024_0_0.201_21172224_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 12:43 _model_00025_0_0.221_21336064_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 12:59 _model_00026_0_0.214_21622784_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 13:34 _model_00027_0_0.212_22237184_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 13:42 _model_00028_0_0.21_22360064_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 13:46 _model_00029_0_0.202_22441984_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 18:27 _model_00030_0_0.212_27357184_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 18:32 _model_00031_0_0.202_27439104_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 18:56 _model_00032_0_0.202_27848704_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 19:01 _model_00033_0_0.201_27930624_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 19:33 _model_00034_0_0.221_28504064_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 19:45 _model_00035_0_0.245_28708864_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 19:50 _model_00036_0_0.216_28790784_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 19:55 _model_00037_0_0.217_28872704_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 20:04 _model_00038_0_0.267_29036544_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 20:16 _model_00039_0_0.245_29241344_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 20:21 _model_00040_0_0.221_29323264_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 20:49 _model_00041_0_0.22_29814784_.zip
-rw-r--r-- 1 flend flend 140927 Mar  2 20:51 _model_00042_0_0.209_29855744_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 01:31 _model_00043_0_0.239_34689024_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 01:33 _model_00044_0_0.248_34729984_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 01:40 _model_00045_0_0.226_34852864_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 01:43 _model_00046_0_0.33_34893824_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 01:47 _model_00047_0_0.212_34975744_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 01:52 _model_00048_0_0.31_35057664_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 01:55 _model_00049_0_0.244_35098624_.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 02:20 _model_00050_0_0.219_35344384_.zip
-rw-r--r-- 1 flend flend 140924 Mar  1 07:05 base.zip
-rw-r--r-- 1 flend flend 140927 Mar  3 02:20 best_model.zip
```

Training speed was also fast (see above)

Performance seems to have saturated at model 6 (5 step sampling) which was after 45 min of training (4 cores).

My hypothesis is that the obvs space of 64 just works better because it's a power of 2. I'll try the teal only cards with the modern code base at 64 obvs space next to see if I can repeat this performance. Otherwise something else is different in the new code base which is killing performance.

## Training run 5 - 7d4f4ab

Returning to the types codebase, using only the first 8 card types and 3 players.

```
ACTIONS = 8
FEATURE_SIZE = 64
```

Trying an observation space of 64 to see if this will give as good results as 079da56 despite the different setup.
If it's not the obvs size I must have broken something else.

We produced 30 models here in a few hours but bizarrely they NEVER win during training (only draw):

![always drawing](./assets/images/alwaysdraw.png)
I don't know if this happens on other runs

The tournament results are dreadful, we have created a model that consistently loses against base. I suspect there is something wrong with the use of random to pick a card when more there is more than 1 of a type.

![results](./assets/images/tournament-7d4f4ab-results.png)

As a comparison, here's a training run with 079da56

![training-079da56](./assets/images/trainingcorrectly.png)

This runs -1 to 1 so this is a problem in 7d4f4ab. The tournament results for 7d4f4ab show that the earlier models do win occasionally against base but they never win against any models (model1 or later).

## Training run 6 - 29664cc

This is as Training run 5 - 7d4f4ab but I fixed a bug in the observation of positions. Honestly I can't see this really affecting anything but I'll give it another go.


```
ACTIONS = 8
FEATURE_SIZE = 64
```

This has the same tapping out at 0.5 issue. But this seems to be that the agents (admittidly I only trained for 10 min or so) always seem to pick invalid actions, causing the game to finish early.

I checked 079da56 and it does not pick illegal actions, even at the start of training.

The issue was that the legal actions need to be at the end of the observation set `obs, legal_actions = split_input(self.processed_obs, ACTIONS)` and they are not in the card_types branch.

## Training run 7 - b59037b

Running the above again with the legal actions fixed.

```
-rw-r--r-- 1 flend flend 95009 Mar 20 15:40 _model_00001_0_0.271_77824_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 17:10 _model_00002_0_0.225_815104_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 17:16 _model_00003_0_0.245_856064_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 20:09 _model_00004_0_0.206_2043904_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 20:23 _model_00005_0_0.201_2084864_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 21:36 _model_00006_0_0.23_2371584_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 21:41 _model_00007_0_0.256_2412544_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 21:46 _model_00008_0_0.238_2453504_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 21:55 _model_00009_0_0.224_2535424_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 22:00 _model_00010_0_0.266_2576384_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 22:54 _model_00011_0_0.214_3108864_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 22:57 _model_00012_0_0.295_3149824_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 23:09 _model_00013_0_0.218_3313664_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 23:15 _model_00014_0_0.332_3395584_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 23:18 _model_00015_0_0.311_3436544_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 23:24 _model_00016_0_0.231_3518464_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 23:27 _model_00017_0_0.289_3559424_.zip
-rw-r--r-- 1 flend flend 95009 Mar 20 23:44 _model_00018_0_0.218_3805184_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 00:08 _model_00019_0_0.203_4132864_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 00:11 _model_00020_0_0.264_4173824_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 00:20 _model_00021_0_0.207_4296704_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 00:23 _model_00022_0_0.305_4337664_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 02:06 _model_00023_0_0.201_5771264_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 02:15 _model_00024_0_0.242_5894144_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 02:18 _model_00025_0_0.211_5935104_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 02:48 _model_00026_0_0.208_6344704_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 02:51 _model_00027_0_0.252_6385664_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 03:12 _model_00028_0_0.238_6672384_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 03:18 _model_00029_0_0.225_6754304_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 03:21 _model_00030_0_0.31_6795264_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 03:47 _model_00031_0_0.265_7163904_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 03:56 _model_00032_0_0.316_7286784_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 04:34 _model_00033_0_0.246_7819264_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 04:37 _model_00034_0_0.229_7860224_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 04:40 _model_00035_0_0.205_7901184_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 04:46 _model_00036_0_0.209_7983104_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 04:49 _model_00037_0_0.245_8024064_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 04:58 _model_00038_0_0.315_8146944_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:01 _model_00039_0_0.281_8187904_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:15 _model_00040_0_0.225_8392704_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:18 _model_00041_0_0.22_8433664_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:24 _model_00042_0_0.256_8515584_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:30 _model_00043_0_0.234_8597504_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:36 _model_00044_0_0.218_8679424_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:42 _model_00045_0_0.221_8761344_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 05:45 _model_00046_0_0.224_8802304_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 06:17 _model_00047_0_0.211_9252864_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 06:23 _model_00048_0_0.236_9334784_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 06:29 _model_00049_0_0.328_9416704_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 06:32 _model_00050_0_0.352_9457664_.zip
-rw-r--r-- 1 flend flend 95009 Mar 21 06:46 _model_00051_0_0.204_9621504_.zip
-rw-r--r-- 1 flend flend 95006 Mar 20 15:32 base.zip
```

Results: viz/card-types-50-model-tournament-results.csv

![training-079da56](./assets/images/card-types-50-model-tournament-results.png)

Results aren't as convincing as previous runs. I played model 36 and it played pretty well, though it's weights showed it hadn't learnt ECL properly and missed a few good opportunities because of that.

## Training run 8 - 490b467

This is identical to training run 7 except that I've added a new selfplay type mostly_best_base which always adds a base player. This is what we test against so probably fair to train against it. It may give better performance by avoiding picking a better model due to random flukes against a very similar model - as we know very similar models tend to block each other all the time.

Results: viz/mostly_base_5.csv-tournament-results.csv

![training-490b467](./assets/images/mostly-base-5.png)

This performed well, perhaps justifying the hypothesis that training models with a performance criterion that we are testing against is a good idea.

## Training run 9 - d814442

This run uses the teal and pink cards (minus FC because it has a card-id that is right at the end of the ID space and I'm not using all the cards yet).

Training died after 12 hours, last model after 10 hours.

```
-rw-r--r-- 1 flend flend 199354 Apr 23 07:34 _model_00001_0_0.2_36864_.zip
-rw-r--r-- 1 flend flend 199354 Apr 23 07:40 _model_00002_0_0.299_118784_.zip
-rw-r--r-- 1 flend flend 199354 Apr 23 07:49 _model_00003_0_0.218_241664_.zip

-rw-r--r-- 1 flend flend 199354 Apr 23 16:56 _model_00049_0_0.215_6672384_.zip
-rw-r--r-- 1 flend flend 199354 Apr 23 16:58 _model_00050_0_0.242_6713344_.zip
-rw-r--r-- 1 flend flend 199354 Apr 23 17:01 _model_00051_0_0.236_6754304_.zip
-rw-r--r-- 1 flend flend 199354 Apr 23 17:04 _model_00052_0_0.221_6795264_.zip
```

Results: viz/teal_and_pink_base_4-tournament-results.csv

Looking at the scores, they seem good. Note that this is a 3-way game (always 1 base) and we are just plotting model0 results, so we expect model0 even when playing against itself (the trace) to get >0 results - because base is strongly negative and the delta is shared between the real models.

I played the best_model and it seemed to play well (it beat me). I wasn't convinced it understood the pink (rare) cards very well but it played a good teal game. It might be interesting to train with a higher ratio of pink cards so they are encountered more often and they are more 'worth' learning.

Tournament with 50 games

![d814442-results](./assets/images/teal-and-pink-4.png)

Looking at the delta between model0 and model1 scores, this (sort of) removes any shared baseline score received from the base model. Therefore this should be 0 on the trace.

![d814442-delta](./assets/images/teal-and-pink-4-score-diff.png)

Doing 500 games to reduce the noise, again plotting the delta

`docker-compose exec app python3 tournament.py -e gonutsfordonuts -st 0 -sp 53 -sx 5 -g 500 -o teal_and_pink_base_4_long > /dev/null`

![d814442-delta-long](./assets/images/teal-and-pink-4-long-score-diff.png)

## Training run 10 - cb045aa

In this training run we are just using the teal cards `teal_deck_filter_no_fc()` but using the same model params as training run 9 to see if a trained teal-and-pink model outperforms a trained teal-only model on teal-and-pink card set.

Results (training teal in pink model): viz/teal_only_in_pink_model-tournament-results.csv

![cb045aa](./assets/images/teal_only_in_pink_model.png)

