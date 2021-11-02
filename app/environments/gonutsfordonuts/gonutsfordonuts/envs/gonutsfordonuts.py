
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutDeckPosition, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered
import gym
import numpy as np
import math

import config

from stable_baselines import logger
from collections import Counter, defaultdict

from .classes import *

class GoNutsScorer:
    # Score turn as if it was the end of the game (so carry out EOG effects)
    @staticmethod
    def score_turn(positions):
        
        player_scores = np.zeros(len(positions))
        
        for p, position in enumerate(positions):
            logger.debug(f'Scoring player: {p}')

            score_dh = GoNutsScorer.score_donut_holes(position)
            logger.debug(f'Donut Holes: {score_dh}')
            player_scores[p] += score_dh
            
            score_gz = GoNutsScorer.score_glazed(position)
            logger.debug(f'Glazed: {score_gz}')
            player_scores[p] += score_gz

            score_jf = GoNutsScorer.score_jelly_filled(position)
            logger.debug(f'Jelly Filled: {score_jf}')
            player_scores[p] += score_jf

            score_fc = GoNutsScorer.score_french_cruller(position)
            logger.debug(f'French Cruller: {score_fc}')            
            player_scores[p] += score_fc

            score_mb = GoNutsScorer.score_maple_bar(position)
            logger.debug(f'Maple Bar: {score_mb}')
            player_scores[p] += score_mb

            score_pwdr = GoNutsScorer.score_powdered(position)
            logger.debug(f'Powdered: {score_pwdr}')
            player_scores[p] += score_pwdr
        
        logger.debug(f'All without plain (all players): {player_scores}')
        score_plain = GoNutsScorer.score_plain(positions)
        logger.debug(f'Plain (all players): {score_plain}')
        player_scores = np.add(player_scores, score_plain)
        logger.debug(f'Final score (all players): {player_scores}')

        return player_scores

    @staticmethod
    def score_plain(positions):
        
        player_scores = np.zeros(len(positions))
        total_plain = []
        
        for p, position in enumerate(positions):
            card_counter = Counter([ c.name for c in position.cards ])
            dn_count = card_counter["plain"]
            player_scores[p] = 1 * dn_count
            total_plain.append((p, dn_count))
        
        max_plain = max(total_plain, key=lambda x: x[1])[1]
        all_max_players = [ x for x in total_plain if x[1] == max_plain ]

        if not max_plain == 0:
            if len(all_max_players) == 1:
                player_scores[all_max_players[0][0]] += 3
            if len(all_max_players) > 1:
                for t in all_max_players:
                    player_scores[t[0]] += 1

        return player_scores

    @staticmethod
    def score_donut_holes(position):
        card_counter = Counter([ c.name for c in position.cards ])
        dh_count = card_counter["donut_holes"]
        if dh_count == 1:
            return 1
        if dh_count == 2:
            return 3
        if dh_count == 3:
            return 6
        if dh_count == 4:
            return 10
        if dh_count == 5:
            return 15

        return 0

    @staticmethod
    def score_jelly_filled(position):
        card_counter = Counter([ c.name for c in position.cards ])
        dn_count = card_counter["jelly_filled"]
        if dn_count == 2 or dn_count == 3:
            return 5
        if dn_count == 4 or dn_count == 5:
            return 10

        return 0

    @staticmethod
    def score_glazed(position):
        card_counter = Counter([ c.name for c in position.cards ])
        dn_count = card_counter["glazed"]
        return dn_count * 2

    @staticmethod
    def score_french_cruller(position):
        card_counter = Counter([ c.name for c in position.cards ])
        dn_count = card_counter["french_cruller"]
        return dn_count * 2

    @staticmethod
    def score_powdered(position):
        card_counter = Counter([ c.name for c in position.cards ])
        dn_count = card_counter["powdered"]
        return dn_count * 3

    @staticmethod
    def score_maple_bar(position):
        card_counter = Counter([ c.name for c in position.cards ])
        types_of_cards = len(card_counter)
        if types_of_cards > 6:
            return card_counter["maple_bar"] * 3
        return 0



class GoNutsGame:

    def __init__(self, n_players):
        self.n_players = n_players

    def setup_game(self, deck_contents=None, shuffle=True):
        self.no_donut_decks = self.n_players + 1
        self.card_types = 9
        
        self.max_score = 200 # to normalise current scores to [0, 1] interval

        if deck_contents:
            self.contents = deck_contents
        else:
            self.contents = self.standard_deck_contents()

        self.total_cards = sum([x['count'] for x in self.contents])

        self.reset_game(shuffle)

    def reset_game(self, shuffle=True):
        self.deck = Deck(self.contents)
        if shuffle:
            self.deck.shuffle()
        
        self.discard = Discard()
        self.players = []
        self.turns_taken = 0
        self.game_ends = False

        player_id = 1
        for p in range(self.n_players):
            self.players.append(Player(str(player_id)))
            player_id += 1

    def standard_deck_contents(self):

        return [
          {'card': ChocolateFrosted, 'info': {}, 'count': 3}  #0 
           ,  {'card': DonutHoles, 'info': {}, 'count':  6} #1 
        ,  {'card': Eclair, 'info': {}, 'count':  3}  #2   
          ,  {'card': Glazed, 'info': {}, 'count':  5} #3  
           ,  {'card': JellyFilled, 'info': {}, 'count':  6} #4 
           ,  {'card': MapleBar,  'info': {}, 'count':  2} #5 
           ,  {'card': Plain, 'info': {}, 'count':  7} #6 
          ,  {'card': Powdered, 'info': {}, 'count':  4}  #7 
          ,  {'card': FrenchCruller, 'info': {}, 'count':  min(self.n_players - 1, 4)}  #8 (last due to variability) 
        ]
    
    def deck_for_card_id(self, card_id):
        for deck in self.donut_decks:
            if deck.card.id == card_id:
                return deck

        logger.debug(f'Cannot find deck for card_id {card_id}')
        raise Exception(f'Cannot find deck for card_id {card_id}')

    def pick_cards(self, cards_ids_picked):
        
        cards_picked = []

        if len(cards_ids_picked) != self.n_players:
            logger.debug('pick_cards() called with wrong number of card_ids')
            raise Exception('pick_cards() called with wrong number of card_ids')

        for i, card_id in enumerate(cards_ids_picked):
        
            card_ids_counter = Counter(cards_ids_picked)

            player = self.players[i]
            deck = self.deck_for_card_id(card_id)

            if card_ids_counter[card_id] > 1:
                deck.set_to_discard()
                cards_picked.append(None)
            else:
                player.position.add_one(deck.card)
                cards_picked.append(deck.card)
                deck.set_taken()
        
        return cards_picked

    def do_card_special_effects(self, cards_picked):
        if len(cards_picked) != self.n_players:
            logger.debug('do_card_special_effects() called with wrong number of card_ids')
            raise Exception('do_card_special_effects() called with wrong number of card_ids')

        for p, card in enumerate(cards_picked):
            if card:
                if card.name == "chocolate_frosted":
                    self.card_action_chocolate_frosted(self, p)
                elif card.name == "eclair":
                    self.card_action_eclair(self, p)

    def card_action_chocolate_frosted(self, player_no):
        # Draw the top card from the draw deck
        print(f'deck size {self.deck.size()}')
        if self.deck.size() > 0:
            self.players[player_no].position.add_one(self.deck.draw_one())

    def card_action_eclair(self, player_no):
        if self.discard.size() > 0:
            self.players[player_no].position.add_one(self.discard.draw_one())

    def reset_turn(self):
        logger.debug(f'\nResetting turn...')
        
        # Setup donut decks for the first time
        if not self.turns_taken:
            self.donut_decks = []
            for i in range (0, self.no_donut_decks):
                self.donut_decks.append(DonutDeckPosition(self.deck.draw_one()))

        # Check end of game
        decks_to_discard = sum(1 for d in self.donut_decks if d.to_discard)
        decks_taken = sum(1 for d in self.donut_decks if d.taken)

        decks_to_refill = decks_to_discard + decks_taken
        if decks_to_refill > self.deck.size():
            self.game_ends = True
        else:
            # Redraw and discard any used decks
            for i in range(0, self.no_donut_decks):
                if self.donut_decks[i].taken:
                    # Already added to player's position
                    logger.debug(f'Filling deck position {i}')
                    self.donut_decks[i] = DonutDeckPosition(self.deck.draw_one())
                elif self.donut_decks[i].to_discard:
                    logger.debug(f'Filling and discarding deck position {i}')
                    self.discard.add([self.donut_decks[i].card])
                    self.donut_decks[i] = DonutDeckPosition(self.deck.draw_one())
                # Otherwise the card stays in position
        
        self.turns_taken += 1
    
    def is_game_over(self):
        return self.game_ends

    def score_turn(self):

        maki = [0] * self.n_players
        
        for i, p in enumerate(self.players):
            count = {'tempura': 0, 'sashimi': 0, 'dumpling': 0}
            for card in p.position.cards:
                if card.type in ('tempura', 'sashimi', 'dumpling'):
                    count[card.type] += 1
                elif card.type == 'maki':
                    maki[i] += card.value
                elif card.type == 'nigiri':
                    if card.played_on_wasabi:
                        p.score += 3 * card.value
                    else:
                        p.score += card.value

            p.score += (count['tempura'] // 2) * 5
            p.score += (count['sashimi'] // 3) * 10
            p.score += min(15, (count['dumpling'] * (count['dumpling'] + 1) ) // 2)
        
        self.score_maki(maki)


class GoNutsForDonutsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(GoNutsForDonutsEnv, self).__init__()
        self.name = 'gonutsfordonuts'
        self.manual = manual

        no_players = 3
        self.n_players = no_players

        self.game = GoNutsGame(no_players)
        self.game.setup_game()

        # player positions / last_choices / discards
        self.total_positions = self.n_players + 2

        # agents choose a card_id as an action.
        # although all card_ids corresponding to the same card type ought to have the same value,
        # it is actual cards we pick and N of them are unmasked each turn
        self.action_space = gym.spaces.Discrete(self.game.total_cards)
        # total positions (see above) / player scores / legal actions
        self.observation_space = gym.spaces.Box(0, 1, (self.game.total_cards * self.total_positions + self.game.n_players + self.action_space.n ,))
        self.verbose = verbose
  
    @property
    def observation(self):
        # Each player may have each individual card
        obs = np.zeros(([self.total_positions, self.game.total_cards]))
        player_num = self.current_player_num

        # Note that tidying the observations to card_id strongly couples the model to the exact
        # composition of the original deck. e.g. increasing the number of FCs due to player count may
        # cause the model to be non-generalisable to different player counts
        for i in range(self.n_players):
            player = self.game.players[player_num]

            # Each player's current position (tableau)
            for card in player.position.cards:
                obs[i][card.id] = 1

            player_num = (player_num + 1) % self.n_players

        # The discard deck
        for card in self.discard.cards:
            obs[self.n_players][card.id] = 1

        # The donut choices picked by the players last time
        # TODO: Make relative to current player
        # Although players pick type, for convenience we observe the card ID they took
        # TODO: Not implemented yet
        if self.turns_taken >= 1:
            for card in self.choices_taken:
                obs[self.n_players + 1][card.id] = 1

        # Current player scores [to guide to which players to target]
        ret = obs.flatten()
        for p in self.game.players: # TODO this should be from reference point of the current_player
            ret = np.append(ret, p.score / self.max_score)

        # Legal actions, representing the donut choices
        ret = np.append(ret, self.legal_actions)

        return ret

    @property
    def legal_actions(self):
        legal_actions = np.zeros(self.action_space.n)

        for i in range(self.game.no_donut_decks):
            legal_actions[self.game.donut_decks[i].card.id] = 1
        
        return legal_actions

    def score_game(self):
        reward = [0.0] * self.n_players
        scores = [p.score for p in self.game.players]
        best_score = max(scores)
        worst_score = min(scores)
        winners = []
        losers = []
        for i, s in enumerate(scores):
            if s == best_score:
                winners.append(i)
            if s == worst_score:
                losers.append(i)

        for w in winners:
            reward[w] += 1.0 / len(winners)
        
        for l in losers:
            reward[l] -= 1.0 / len(losers)

        return reward

    @property
    def current_player(self):
        return self.game.players[self.current_player_num]

    def step(self, action):
        
        reward = [0] * self.n_players
        done = False

        # illegal action
        if self.legal_actions[action] == 0:
            reward = [1.0/(self.n_players-1)] * self.n_players
            logger.debug(f'Illegal action played {action}')
            reward[self.current_player_num] = -1
            # I think this jettisons the run
            done = True

        # vote for a card; pick cards if all players have voted
        else:

            # TODO: For modelling discard actions etc. we need to use a state machine here
            # Currently only the 'pick-a-donut' state is modelled
            # The actions are only: 'pick-card-id'

            self.action_bank.append(action)

            if len(self.action_bank) == self.n_players:
                logger.debug(f'\nThe chosen cards are now competitively picked')

                cards_picked = self.game.pick_cards(self.action_bank)
                # TODO: Move to a new step in the state machine
                self.game.do_card_special_effects(cards_picked)
                self.game.reset_turn()

                # Per-turn scores are an observation for the agents
                self.game.score_turn()

                self.action_bank = []
            
            self.current_player_num = (self.current_player_num + 1) % self.n_players

            # Check end-of-game condition (no donuts less than no of spaces)
            if self.game.is_game_over():
                reward = self.score_game()
                done = True
            else:
                self.render()

        self.done = done

        return self.observation, reward, done, {}

    def reset(self):

        self.game.reset_game()

        self.action_bank = []

        self.current_player_num = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False):
        
        if close:
            return

        if self.turns_taken < self.cards_per_player:
            logger.debug(f'\n\n-------ROUND {self.round} : TURN {self.turns_taken + 1}-----------')
            logger.debug(f"It is Player {self.current_player.id}'s turn to choose")
        else:
            logger.debug(f'\n\n-------FINAL ROUND {self.round} POSITION-----------')
            

        for p in self.players:
            logger.debug(f'\nPlayer {p.id}\'s hand')
            if p.hand.size() > 0:
                logger.debug('  '.join([ str(card.order) + ': ' + card.symbol for card in sorted(p.hand.cards, key=lambda x: x.id)]))
            else:
                logger.debug('Empty')

            logger.debug(f'Player {p.id}\'s position')
            if p.position.size() > 0:
                logger.debug('  '.join([str(card.order) + ': ' + card.symbol + ': ' + str(card.id) for card in sorted(p.position.cards, key=lambda x: x.id)]))
            else:
                logger.debug('Empty')

        logger.debug(f'\n{self.deck.size()} cards left in deck')
        logger.debug(f'{self.discard.size()} cards discarded')

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
            

        if self.turns_taken == self.cards_per_player:
            for p in self.players:
                logger.debug(f'Player {p.id} points: {p.score}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Go Nuts For Donuts!')
