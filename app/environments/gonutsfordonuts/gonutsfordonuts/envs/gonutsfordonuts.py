
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
        
        logger.info(f'All without plain (all players): {player_scores}')
        score_plain = GoNutsScorer.score_plain(positions)
        logger.info(f'Plain (all players): {score_plain}')
        player_scores = np.add(player_scores, score_plain)
        logger.info(f'Final score (all players): {player_scores}')

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


class GoNutsGameGymTranslator:

    def __init__(self, donuts_game):
        self.game = donuts_game

        # Define the maximum card space for the max numbers of players
        # (for lower number of player games, non-player observations are zeroed)
        self.total_possible_cards = 70
        self.total_possible_players = 5

    def total_positions(self):
        # player positions / discards
        return self.game.n_players + 1

    def observation_space_size(self):

        # player positions / discard / player scores / legal actions
        return self.total_possible_cards * self.total_possible_players + self.total_possible_cards + self.total_possible_players + self.action_space_size()

    def action_space_size(self):
        # agents choose a card_id as an action.
        # although all card_ids corresponding to the same card type ought to have the same value,
        # it is actual cards we pick and N of them are unmasked each turn
        return self.total_possible_cards
    
    def get_legal_actions(self):
        
        legal_actions = np.zeros(self.action_space_size())

        for i in range(self.game.no_donut_decks):
            legal_actions[self.game.donut_decks[i].card.id] = 1
        
        return legal_actions

    def get_observations(self, current_player_num):

        # To make this a generalisable model
        # Always assume we are playing with 5 players and just 0 out their observations
        # Always assume we are playing with all the cards, cards we are not playing with will not occur

        n_players = self.game.n_players

        obs = np.zeros([self.total_possible_players, self.total_possible_cards])

        # Each player's current position (tableau)
        # starting from the current player and cycling to higher-numbered players
        player_num = current_player_num
        for i in range(n_players):
            player = self.game.players[player_num]

            for card in player.position.cards:
                obs[i][card.id] = 1

            player_num = (player_num + 1) % n_players

        ret = obs.flatten()

        # The discard deck
        # Just the top card of the discard pile
        discard = np.zeros(self.total_possible_cards)
        #for i, card in enumerate(list(reversed(self.game.discard.cards))[:self.discard_size]):
        if self.game.discard.size():
            discard[self.game.discard.peek_one().id] = 1

        ret = np.append(ret, discard)
        
        # Current player scores [to guide the agent to which players to target]
        player_scores = np.zeros(self.total_possible_players)

        player_num = current_player_num
        for i in range(n_players):
            player = self.game.players[player_num]
            player_scores[i] = player.score / self.game.max_score
            player_num = (player_num + 1) % n_players

        ret = np.append(ret, player_scores)

        # Legal actions, representing the donut choices
        ret = np.append(ret, self.get_legal_actions())

        return ret

class GoNutsGame:

    def __init__(self, n_players):
        self.n_players = n_players

    def setup_game(self, shuffle=True, deck_order=None, deck_filter=None):
        self.no_donut_decks = self.n_players + 1
        self.card_types = 9
        
        self.max_score = 200 # to normalise current scores to [0, 1] interval

        self.contents = GoNutsGame.standard_deck_contents()

        self.total_cards = sum([x['count'] for x in self.contents])

        self.reset_game(shuffle=shuffle, deck_order=deck_order, deck_filter=deck_filter)

    def reset_game(self, shuffle=True, deck_order=None, deck_filter=None):
        self.deck = Deck(self.n_players, GoNutsGame.standard_deck_contents())
        if shuffle:
            self.deck.shuffle()
        elif deck_order:
            self.deck.reorder(deck_order)
        elif deck_filter:
            self.deck.filter(deck_filter)
        
        self.discard = Discard()
        self.players = []
        self.turns_taken = 0
        self.game_ends = False

        player_id = 1
        for p in range(self.n_players):
            self.players.append(Player(str(player_id)))
            player_id += 1

    def start_game(self):
        # Setup donut decks for the first time
        self.donut_decks = []
        for i in range (0, self.no_donut_decks):
            self.donut_decks.append(DonutDeckPosition(self.deck.draw_one()))

    @classmethod
    def standard_deck_contents(self):

        # # 36 + (np - 1) cards
        # return [
        #   {'card': ChocolateFrosted, 'info': {}, 'count': 3}  #0 
        #    ,  {'card': DonutHoles, 'info': {}, 'count':  6} #1 
        # ,  {'card': Eclair, 'info': {}, 'count':  3}  #2   
        #   ,  {'card': Glazed, 'info': {}, 'count':  5} #3  
        #    ,  {'card': JellyFilled, 'info': {}, 'count':  6} #4 
        #    ,  {'card': MapleBar,  'info': {}, 'count':  2} #5 
        #    ,  {'card': Plain, 'info': {}, 'count':  7} #6 
        #   ,  {'card': Powdered, 'info': {}, 'count':  4}  #7 
        #   ,  {'card': FrenchCruller, 'info': {}, 'count':  min(self.n_players - 1, 4)}  #8 (last due to variability) 
        # ]

        # Full standard deck contents
        # 66 + (np - 1) cards [define as 5 - 1 = 4 in observation space] = 70 cards
        # Note: reversed so dealt in top-to-bottom order
        standard_deck_contents = [ {'card': ChocolateFrosted, 'info': {}, 'count': 3}  #0 
           ,  {'card': DonutHoles, 'info': {}, 'count':  6} #1 
           ,  {'card': Eclair, 'info': {}, 'count':  3}  #2   
           ,  {'card': Glazed, 'info': {}, 'count':  5} #3  
           ,  {'card': JellyFilled, 'info': {}, 'count':  6} #4 
           ,  {'card': MapleBar,  'info': {}, 'count':  2} #5 
           ,  {'card': Plain, 'info': {}, 'count':  7} #6 
           ,  {'card': Powdered, 'info': {}, 'count':  4}  #7         
           ,  {'card': BostonCream, 'info': {}, 'count':  6} #8
           ,  {'card': DoubleChocolate, 'info': {}, 'count':  2} #9
           ,  {'card': RedVelvet, 'info': {}, 'count':  2} #10
           ,  {'card': Sprinkled, 'info': {}, 'count':  2} #11
           ,  {'card': BearClaw, 'info': {}, 'count':  2} #12
           ,  {'card': CinnamonTwist, 'info': {}, 'count':  2} #13
           ,  {'card': Coffee, 'info': {}, 'count':  2} #14
           ,  {'card': DayOldDonuts, 'info': {}, 'count':  1} #15
           ,  {'card': Milk, 'info': {}, 'count':  1} #16
           ,  {'card': OldFashioned, 'info': {}, 'count':  2} #17
           ,  {'card': MapleFrosted, 'info': {}, 'count':  2} #18
           ,  {'card': MuchoMatcha, 'info': {}, 'count':  2} #19
           ,  {'card': RaspberryFrosted, 'info': {}, 'count':  2} #19
           ,  {'card': StrawberryGlazed, 'info': {}, 'count':  2} #20
           ,  {'card': FrenchCruller, 'info': {}, 'count':  4}  #21 (last due to variability) 
        ]

        return standard_deck_contents
    
    def deck_for_card_id(self, card_id):
        for deck in self.donut_decks:
            if deck.card.id == card_id:
                return deck

        logger.info(f'Cannot find deck for card_id {card_id}')
        raise Exception(f'Cannot find deck for card_id {card_id}')

    def pick_cards(self, cards_ids_picked):
        
        cards_picked = []

        if len(cards_ids_picked) != self.n_players:
            logger.info('pick_cards() called with wrong number of card_ids')
            raise Exception('pick_cards() called with wrong number of card_ids')

        for i, card_id in enumerate(cards_ids_picked):
        
            card_ids_counter = Counter(cards_ids_picked)

            player = self.players[i]
            deck = self.deck_for_card_id(card_id)

            if card_ids_counter[card_id] > 1:
                deck.set_to_discard()
                logger.debug(f'Discarding {deck.card.symbol}')
                cards_picked.append(None)
            else:
                player.position.add_one(deck.card)
                logger.debug(f'Player {player.id} picks {deck.card.symbol}')

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
                    self.card_action_chocolate_frosted(p)
                elif card.name == "eclair":
                    self.card_action_eclair(p)

    def card_action_chocolate_frosted(self, player_no):
        # Draw the top card from the draw deck
        print(f'deck size {self.deck.size()}')
        if self.deck.size() > 0:
            self.players[player_no].position.add_one(self.deck.draw_one())

    def card_action_eclair(self, player_no):
        if self.discard.size() > 0:
            self.players[player_no].position.add_one(self.discard.draw_one())

    def reset_turn(self):

        self.turns_taken += 1

        logger.info(f'\nSetting up turn {self.turns_taken}...')
        
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
                    self.donut_decks[i] = DonutDeckPosition(self.deck.draw_one())
                    logger.info(f'Filling deck position {i} with card {self.donut_decks[i].card.symbol}')

                elif self.donut_decks[i].to_discard:
                    discarded_card = self.donut_decks[i].card
                    self.discard.add([discarded_card])
                    self.donut_decks[i] = DonutDeckPosition(self.deck.draw_one())
                    logger.info(f'Discarding {discarded_card.symbol} from deck position {i} and filling with card {self.donut_decks[i].card.symbol}')

                # Otherwise the card stays in position
                else:
                    logger.info(f'Deck position {i} stays unchanged with card {self.donut_decks[i].card.symbol}')
                    
    def is_game_over(self):
        return self.game_ends

    def score_turn(self):

        player_scores = GoNutsScorer.score_turn([ p.position for p in self.players ])
        
        for i, p in enumerate(self.players):
            p.score = player_scores[i]

    def record_player_actions(self, player_card_picks):
        self.last_player_card_picks = list(player_card_picks)

    def do_player_actions(self, player_card_picks):

        logger.info(f'\nThe chosen cards are now competitively picked')

        self.record_player_actions(player_card_picks)

        cards_picked = self.pick_cards(player_card_picks)
        # TODO: Move to a new step in the state machine
        self.do_card_special_effects(cards_picked)
        self.reset_turn()

        # Per-turn scores are an observation for the agents
        self.score_turn()

    def player_scores(self):
        return [ p.score for p in self.players ]

class GoNutsForDonutsEnvUtility:

    @staticmethod
    def score_game_from_players(players):

        reward = [0.0] * len(players)
        scores = [p.score for p in players]
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
        self.game.start_game()

        self.translator = GoNutsGameGymTranslator(self.game)

        self.action_space = gym.spaces.Discrete(self.translator.action_space_size())
        self.observation_space = gym.spaces.Box(0, 1, (self.translator.observation_space_size(),))
        self.verbose = verbose
  
    @property
    def observation(self):
        return self.translator.get_observations(self.current_player_num)

    @property
    def legal_actions(self):
        return self.translator.get_legal_actions()

    def score_game(self):
        return GoNutsForDonutsEnvUtility.score_game_from_players(self.game.players)

    @property
    def current_player(self):
        return self.game.players[self.current_player_num]

    def step(self, action):
        
        reward = [0] * self.n_players
        done = False

        # illegal action
        if self.legal_actions[action] == 0:
            reward = [1.0/(self.n_players-1)] * self.n_players
            logger.info(f'Illegal action played {action}')
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

                self.game.do_player_actions(self.action_bank)
                self.action_bank = []
            
            self.current_player_num = (self.current_player_num + 1) % self.n_players

            # Check end-of-game condition (no donuts less than no of spaces)
            if self.game.is_game_over():
                reward = self.score_game()
                done = True
            else:
                pass #self.render()

        self.done = done

        return self.observation, reward, done, {}

    def reset(self):

        self.game.reset_game()
        self.game.start_game()

        self.action_bank = []

        self.current_player_num = 0
        self.done = False
        logger.info(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False):
        
        if close:
            return

        print(f'\n\n-------TURN {self.game.turns_taken + 1}-----------')
        print(f"It is Player {self.current_player.id}'s turn to choose")            

        # Render player positions

        for p in self.game.players:
            print(f'Player {p.id}\'s position')
            if p.position.size() > 0:
                print('  '.join([str(card.order) + ': ' + card.symbol + ': ' + str(card.id) for card in sorted(p.position.cards, key=lambda x: x.id)]))
            else:
                print('Empty')

        # Render donuts to choose
        for i, d in enumerate(self.game.donut_decks):
            this_card = d.get_card()
            print(f'Deck {i}: {this_card.symbol}; {this_card.id}')

        # Top of discard
        if self.game.discard.size():
            print(f'Discard: {self.game.discard.size()} cards, top {self.game.discard.peek_one().symbol}')

        print(f'\n{self.game.deck.size()} cards left in deck')

        if self.verbose:
            print(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            legal_action_str = "Legal actions: "

            for i,o in enumerate(self.legal_actions):
                if o:
                    card_for_action = next((c for c in self.game.deck.base_deck if c.id == i), None)
                    if card_for_action:
                        legal_action_str += f"{i}:{card_for_action.symbol} "
                    else:
                        print(f"Can't find card for action {o}")
            print(legal_action_str)

        if self.done:
            print(f'\n\nGAME OVER')
            
        for p in self.game.players:
            print(f'Player {p.id} points: {p.score}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Go Nuts For Donuts!')
