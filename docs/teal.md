# Reinforcement learning experiment with Go Nuts For Donuts
## Introduction
I decide to have a go at reinforcement learning (RL). It’s the exciting but less-useful (so far) cousin of the conventional machine learning techniques that allow us to tell cats from dogs, let cars drive by themselves and lets Google understand you. RL allows you to build models that explore environments and build behaviours based on a reward. You don’t need a set of ground truths (like supervised learning) or a full probabilistic modelling of the environment (like dynamic programming). I think of RL learning a bit like we do as humans — building on behaviours that reach our goal and discarding those that we find unhelpful.

Needing to start with something simple, I turned to Go Nuts For Donuts — a boardgame which sits downstairs and my 4-year old daughter enjoys ‘playing’. You can find the rules here (at time of writing). I recommend you pick up a copy — it’s inexpensive and supports the designer and publisher. Simple board games should form good environments for RL — they have small(ish) discrete action spaces and the game logic that the agents experience is easy(ish) to write.

Donuts (as I shall now call it) is a straightforward game where a number (let’s say 4) donut cards are placed out in a row and all players pick one in secret. The choices are revealed — any donut picked by only one player becomes that player’s; any donut picked by two or more players is discarded. Some donuts have special rules — passing a donut to another player etc. The row of donuts is then refilled. At the end of the game you count up the number of points you have — some donuts score more than others.

## Implementation
I was excited to discover the simple repo on github (blog post) — a toolkit for building RL agents for boardgames. It uses the ‘self play’ method that was so effective for AlphaGo Zero — an agent plays against earlier versions of itself to continually improve. It uses the PPO algorithm from OpenAI to train the agent, as implemented in the (old) stable-baselines. So I forked the repo and did the tedious devops stuff needed to get it working with python 3.7 which the latest ubuntu Docker image seemed to want etc.

To start, I implemented the 2-player version of Donuts (see rules) which only uses the simplest (teal) cards although I went for 3 players. I don’t think it’s a terribly entertaining version of the game but I figured start simple. Implementing the game logic was straightforward and I unit tested it extensively, figuring better to find bugs before the agent got cracking learning its way around (or exploiting!) them.

In the self-play environment, there are 3 key piece of data to produce / extract from the environment (the traditional game logic). These three being the observation space, action space and rewards.

For rewards, I copied the simple repo’s standard reward function which broadly splits 1.0 amongst the winners (highest scores) and -1.0 amongst the losers (lowest scores) and 0 for everyone in between. This is awarded at the end of the game once all players have finished. Although the sparseness of having no reward during the game itself is a problem for complex environments, for a short game like Donuts I figured it’d be fine.

The action space represents the actions the agents can take on any turn. In my simple version of Donuts, these actions are limited to taking a card out of one of the four positions. An example layout is shown below:

<img src="docs/images/donut_positions.png" alt="hi" class="inline"/>

An agent playing at this state has 4 choices with this deal:
 - (Pos 1, Donut Holes)
 - (Pos 2, Glazed)
 - (Pos 3, Donut Holes)
 - (Pos 4, Eclair)

In this simple version of the game there are 9 types of cards and each could appear in any of the 4 positions. That means there’s 36 theoretical possible actions in total:

[(Pos 1, Donut Holes), (Pos 2, Donut Holes), (Pos 3, Donut Holes), (Pos 4, Donut Holes), (Pos 1, Glazed), (Pos 2, Glazed)…etc.etc.]

Of course, each turn there are actually only 4 of these actions available to any agent — the donuts that were actually dealt. We ensure the agents don’t take any of the impossible actions by masking — setting the probability of those actions (that comes out of the policy head of the neural network — see original blogpost) to zero.

But, if you think about it, there’s not much difference between picking the Donut Holes in Pos 1 above over the one in Pos 3. Well, unless one of your opponents consistently likes low numbers. Whilst this is a possibility (and could reasonably be learnt to be exploited by an agent provided the observations [later] included the memory of which position each card had been taken from) it’s unlikely in self-play that any agent that preferred a particular position would be maintained as part of the self-play. Therefore it would likely be better to reduce this action space to just the 9 unique card types and get each agent to pick one of the cards at random if multiples appeared on the board.

Anyway, if I’d been sensible I’d have done that — instead I set my actions space to be each of the 41 unique cards in the deck. This means the agents learn a probability for taking each individual copy of, say, Glazed, in each particular state (the agents don’t know what the cards are a-priori and learn this from play). From examining the agent’s policy, unsurprisingly, the probability of taking each, say, Glazed is very close to each other. Effectively the agent has learnt that each Glazed is the same, despite my suboptimal choice of action space. This choice of action space, as well as being slow to learn, is also highly fragile to increasing player number, since in Donuts certain cards (FC) scale with the number of players. The only advantage is that it naturally copes with multiple copies of the same card type being dealt. Nonetheless, it works for the setup being modelled.

As I final note I arbitrarily ruled that my agents wouldn’t use the French Cruller special ability since that would require a larger action space — this will return when I model the other cards.

The observation space should comprise what you think the agents ought to know in order to play the game successfully. In this case I concatenated the following useful things together:

- The cards each player has won so far (in my modelling, 0 or 1 for the 38 cards in the deck for each of the 3 players)
- The contents of the discard deck (0/1 for 38 cards) — useful for deciding if the eclair (take top of discard pile) is a strong play. I notice now that this should actually be a stack (ordered) and probably doesn’t need to contact more than the top 2 or 3 cards.
- The donut choices the players took in the last round (3 card ids) — although I doubt this is very useful when you have the won cards
- Each player’s score, dividing into a 0–1 interval — I imagine this might be useful to decide which player to block or target if the agents play well enough
- The legal actions for this round (0/1 for 38 cards)

On reflection, given the simplicity of this version of the game (and the results below), it may be that only the legal actions are really required for good play. However, other players’ positions and scores will likely be important when I introduce cards that allow you to pass a (negative) card to a player etc.

PPO uses a neural network architecture to estimate the value function (how much a move will be worth from any state) and a policy (the probability of the agent taking any move from the same state). I kept the model the same as the one used for SushiGO! in the original blogpost, bar changing the size of the size of the final output of the policy head to the correct number of actions (38).

## Results
The agent was trained with 3 threads on my i5–4690K with no GPU acceleration for about a day. Training parameters were left at the defaults, including the reward threshold of +0.2 required to create a new generation of (hopefully improved) agent after 100? plays.

That produced 126 generations of models over the day although, as we shall see, they didn’t seem to get much stronger after generation 2 that was trained after about 5 minutes.

To evaluate the agent, I played against it in a 3-way match up, the best agent vs. me vs. the base agent (random play). I set the agents to always do their best plays (highest probability from the policy) rather than using the probabilities to pick a card. To explain, if the agent’s policy produced probabilities [0.5, 0.25, 0.25, 0] in ‘best play’ mode the agent would always pick option 1, whereas in ‘normal’ mode it would pick option 1 50% of the time.

It was clear (from play and looking the policy probabilities) that the agent had learnt the value of cards and made good plays, including using the eclair to take good cards from the discard pile. It played roughly as strongly as I did. However, I noticed a funny thing — since myself and the trained agent (in best play mode) kept going for the best cards, we kept blocking each other, getting no cards and the random play agent consistently won despite knowing nothing about the game! It reminded me a bit of playing with another adult and a young child.

To test this further, I set up a tournament as described in the original blogpost. Unfortunately code was not provided, so I added a tournament play file and figure generator.

As mentioned above, the model was trained for 3-person play so I set up a tournament with 3 players (model1, model2, base) where model1 and model2 were sampled from increasing generations of trained models and the base was random play.

With model1 and model2 picking their best actions from the policy, the tournament results are as follows. The number in each cell is the mean reward for model1 across 50 games, with 1.00 being winning every time and -1.00 being losing every time.

<img src="app/blog/blog1_reinforcement_learning_expt/1_10_best_action.png" alt="hi" class="inline"/>

Figure 1: 3–way tournament picking best actions between (model1, model2, base). model1 mean reward over 50 games.

On the far left we see that the trained models 1 through 10 almost always win against two base agents (far left) and models 2 and up almost always win against (model1, base). Whilst models 5 and up typically win against lower models and base, the margin of victory drops quickly to 0.1–0.2 (reds). Once two good models play each other (models 5 and up, bottom right of diagrams) the results are noisy but often go negative. This is where the good agents start blocking each other allowing the base (random element) to win. This is also seen if we plot the mean reward of the base agent during the same tournament.

<img src="app/blog/blog1_reinforcement_learning_expt/1_10_best_action_base_agent.png" alt="hi" class="inline"/>

Figure 2: 3–way tournament picking best actions [same results as Figure 1] between (model1, model2, base). base mean reward over 50 games.

The base (random) model does best in the bottom-right of the image with stronger agents and interesting does best of all down the diagonal. The diagonal represents the same agent playing against itself where it is excellent at blocking and therefore the random player wins by simply taking non-optimal cards the other players don’t want. The checkerboard pattern is interesting, perhaps suggesting that there is similarity between the play of even and odd models.

Now, it’s slightly surprising that the agents have been trained in self-play in a three-way environment to let a random agent win. However:

- In the above tournament we forced the agents to always take their highest probability action whereas during training they stochastically play via the probabilities from the policy.
- During training the models were trained with ‘mostly-best’ opponents, i.e. they didn’t see poor opponents very often

If we relax the requirement that models always pick the best action and instead stochastically pick actions from policy, the additional randomness (or perhaps reflecting the model training) gives better results for the trained models

<img src="app/blog/blog1_reinforcement_learning_expt/1_10_non_best_action.png" alt="hi" class="inline"/>

Figure 3: 3–way tournament stochastically picking actions between (model1, model2, base). model1 mean reward over 50 games.
The bottom-right of the figure stays mostly positive showing that even when two good agents are fighting they do better than random play (although not by very much!). The variance in their actions means they block less and get make a position.
Does the picture change with better trained agents? Not really, see the same view for a sample of models up to the latest trained.

Figure 3: 3–way tournament stochastically picking actions between (model1, model2, base). model1 mean reward over 50 games. Up to model generation 121.

It’s hard to saw that any model past model 13 has better performance in the tournament setting against a similar agent and random agent.

## Conclusions

So that’s the story so far! We have trained a decent agent for playing the (very simple) 2-player card set of Go Nuts for Donuts that competes well against a human (me). We seem to have reached a limit in the strength of the model very quickly (5 or so generations). Since the game is so simple, perhaps that is the best play achievable or perhaps the agent hasn’t discovered something (that I also don’t know!).

Next steps — simplify the action space as we discussed above and add more complex and interesting cards into the deck. This will require new actions to model choosing cards to discard, transfer to other players etc. and it will be interesting to see if different strategies emerge.
