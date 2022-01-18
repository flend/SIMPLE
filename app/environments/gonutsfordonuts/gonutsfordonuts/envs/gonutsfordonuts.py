
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutDeckPosition, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered
import gym
import numpy as np
import math
import gonutsfordonuts.envs.cards as cards
import gonutsfordonuts.envs.obvs as obvs

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

        positions = np.zeros([self.total_possible_players, self.total_possible_cards])

        # Each player's current position (tableau)
        # starting from the current player and cycling to higher-numbered players

        # create from player 0 perspective
        for i in range(n_players):
            player = self.game.players[i]

            for card in player.position.cards:
                positions[i][card.id] = 1

        positions_flat = positions.flatten()
        # roll forward (+wrap) to put this player's numbers at the start
        positions_rolled = np.roll(positions_flat, (self.total_possible_players - current_player_num) * self.total_possible_cards)
        ret = positions_rolled

        # The discard deck
        discard = np.zeros(self.total_possible_cards)

        for card in self.game.discard.cards:
            discard[card.id] = 1
        
        ret = np.append(ret, discard)

        # The top discard card [for eclair]
        top_discard = np.zeros(self.total_possible_cards)

        if self.game.discard.size():
            top_discard[self.game.discard.peek_one().id] = 1
        
        ret = np.append(ret, top_discard)

        # Current player scores [to guide the agent to which players to target]
        player_scores = np.zeros(self.total_possible_players)

        for i in range(n_players):
            player = self.game.players[i]
            player_scores[i] = player.score / self.game.max_score

        scores_rolled = np.roll(player_scores, self.total_possible_players - current_player_num)

        ret = np.append(ret, scores_rolled)

        # Legal actions, representing the donut choices
        ret = np.append(ret, self.get_legal_actions())

        return ret

class GoNutsGameState:
    PICK_DONUT = 0
    PICK_DISCARD = 1
    INSTANT_ACTION = 2

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
        
        if deck_order:
            self.deck.reorder(deck_order)
        if deck_filter:
            self.deck.filter(deck_filter)        
        if shuffle:
            self.deck.shuffle()
         
        self.discard = Discard()
        self.players = []
        self.turns_taken = 0
        self.game_ends = False
        self.donut_picks_action_bank = []
        self.donut_player = 0

        self.game_state = GoNutsGameState.PICK_DONUT

        player_id = 0
        for p in range(self.n_players):
            self.players.append(Player(str(player_id)))
            player_id += 1

    def start_game(self):
        # Setup donut decks for the first time
        self.donut_decks = []
        for i in range (0, self.no_donut_decks):
            self.donut_decks.append(DonutDeckPosition(self.deck.draw_one()))

    @classmethod
    def teal_deck_filter(self):
        return [ cards.CF_FIRST, cards.CF_2, cards.CF_3, cards.DH_FIRST, cards.DH_2, cards.DH_3, cards.DH_4, cards.DH_5, cards.DH_6,
        cards.ECL_FIRST, cards.ECL_2, cards.ECL_3, cards.GZ_FIRST, cards.GZ_2, cards.GZ_3, cards.GZ_4, cards.GZ_5,
        cards.JF_FIRST, cards.JF_2, cards.JF_3, cards.JF_4, cards.JF_5, cards.JF_6, cards.MB_FIRST, cards.MB_2,
        cards.P_FIRST, cards.P_2, cards.P_3, cards.P_4, cards.P_5, cards.P_6, cards.P_7,
        cards.POW_FIRST, cards.POW_2, cards.POW_3, cards.POW_4, cards.FC_FIRST, cards.FC_2 ]


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

    def discard_card_for_card_id(self, card_id):
        for card in self.discard:
            if card.id == card_id:
                return card

        logger.info(f'Cannot find card_id {card_id} in discard')
        raise Exception(f'Cannot find card_id {card_id} in discard')

    def pick_cards(self, cards_ids_picked):
        
        if not self.game_state == GoNutsGameState.PICK_DONUT:
            logger.error(f'pick_cards() called in incorrect state {self.game_state}')

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
                logger.info(f'Discarding {deck.card.symbol}')
                cards_picked.append(None)
            else:
                player.position.add_one(deck.card)
                logger.info(f'Player {player.id} picks {deck.card.symbol}')

                cards_picked.append(deck.card)
                deck.set_taken()
        
        return cards_picked

    def do_immediate_card_special_effects(self, player_no, card):
        """Do card special effects that don't need a game state change"""

        if card:
            
            if card.name == "chocolate_frosted":
                self.card_action_chocolate_frosted(player_no)
            elif card.name == "eclair":
                self.card_action_eclair(player_no)
            else:
                logger.info(f"No special effect for card {card.symbol} for player {player_no}")


    def card_action_chocolate_frosted(self, player_no):
        # Draw the top card from the draw deck
        logger.info(f"Card action Chocolate Frosted (draw one from deck) for player {player_no}")
        if self.deck.size() > 0:
            logger.debug(f"Adding {self.deck.peek_one().symbol} to position of {player_no}")
            self.players[player_no].position.add_one(self.deck.draw_one())

    def card_action_eclair(self, player_no):
        logger.info(f"Card action Eclair (draw top from discard) for player {player_no}")

        if self.discard.size() > 0:
            logger.debug(f"Adding {self.discard.peek_one().symbol} to position of {player_no}")
            self.players[player_no].position.add_one(self.discard.draw_one())

    def reset_turn(self):

        self.donut_picks_action_bank = []

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

    def do_pick_cards_action(self, player_card_picks):

        logger.info(f'\nThe chosen cards are now competitively picked')
        self.record_player_actions(player_card_picks)
        self.cards_picked = self.pick_cards(player_card_picks)

    def do_pick_discard_action(self, player_no, player_discard_pick):
        logger.info(f"Card action Red Velvet (draw card from discard) for player {player_no}")

        discard_card_to_pick = self.discard_card_for_card_id(player_discard_pick)

        logger.debug(f"Adding {discard_card_to_pick.symbol} to position of {player_no}")
        self.players[player_no].position.add_one(discard_card_to_pick)
        self.discard.remove_one(player_discard_pick)

    def do_end_turn_after_all_player_actions(self):

        self.reset_turn()

        # Per-turn scores are an observation for the agents
        self.score_turn()

    def player_scores(self):
        return [ p.score for p in self.players ]

    def game_state_is_non_step_action(self):

        if self.game_state == GoNutsGameState.INSTANT_ACTION:
            return True
        
        # Other actions require the AI to make a choice, DONUT, DISCARD etc.
        return False

    def execute_game_loop(self, step_action):

        next_player_no = self.execute_game_state_action(step_action)

        # If we are now in a state that doesn't require a step, keep looping until we need a step
        while self.game_state_is_non_step_action():
            next_player_no = self.execute_game_state_action(None)
        
        # Return the player number that is next required to do a step (could be getting a donut or some other action)
        return next_player_no

    def move_to_next_action_player(self):
        self.action_player += 1

    def check_action_for_this_action_player_and_set_state(self):

        # Instant actions (including no-op for cards that have no action)
        new_state = GoNutsGameState.INSTANT_ACTION

        # All players have performed their card actions, leave action state and go back to picking donuts
        if self.action_player == self.n_players:
            self.game_state = GoNutsGameState.PICK_DONUT
            self.donut_player = 0
            self.action_player = 0
            logger.info("Resetting turn after all player actions")
            self.do_end_turn_after_all_player_actions()
            return self.donut_player
        
        # Examine donut pick for this player and set state
        card = self.cards_picked[self.action_player]
        if card:
            if card.name == "red_velvet":
                new_state = GoNutsGameState.PICK_DISCARD

        self.game_state = new_state
        return self.action_player

    def execute_game_loop_with_actions(self, step_actions):
        """Used in testing"""
        for action in step_actions:
            self.execute_game_loop(action)

    def execute_game_state_action(self, step_action):
        
        # CHECK ACTION STATES

        # Do instant actions - does not require step, action state
        logger.debug(f'Game state: {repr(self.game_state)}')
        if self.game_state == GoNutsGameState.INSTANT_ACTION:
            self.do_immediate_card_special_effects(self.action_player, self.cards_picked[self.action_player])
            self.move_to_next_action_player()
            return self.check_action_for_this_action_player_and_set_state() # will set back to DONUT state if all actions complete
        
        # Discard action - requires step, action state
        elif self.game_state == GoNutsGameState.PICK_DISCARD:

            discard_action = GoNutsGame.translate_step_action(self.game_state, step_action)
            self.do_pick_discard_action(self.action_player(), discard_action)
            # Move to the next player to have an action
            self.move_to_next_action_player()
            return self.check_action_for_this_action_player_and_set_state()

        # CHECK DONUT STATES

        # Do player picks card - requires step, donut state
        elif self.game_state == GoNutsGameState.PICK_DONUT:

            donut_action = GoNutsGame.translate_step_action(self.game_state, step_action)
            self.donut_picks_action_bank.append(donut_action)

            # All players have picked a card, process actions
            if len(self.donut_picks_action_bank) == self.n_players:

                self.do_pick_cards_action(self.donut_picks_action_bank)
                self.donut_picks_action_bank = []
                self.action_player = 0
                return self.check_action_for_this_action_player_and_set_state()

            else:
                self.donut_player += 1
                return self.donut_player     
            
        logger.error(f"Unknown game state {self.game_state}")
        raise RuntimeError(f"Unknown game state {self.game_state}")
    
    @classmethod
    def translate_step_action(cls, game_state, step_action):
        if game_state == GoNutsGameState.PICK_DONUT:
            return step_action - obvs.OBVS_POSITIONS_START
        elif game_state == GoNutsGameState.PICK_DISCARD:
            return step_action - obvs.OBVS_ALL_DISCARD_START
        else:
            logger.error(f"Can't translate action {step_action} from game state {game_state}")
            raise RuntimeError(f"Unknown game state {game_state}")
    


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

        # do actions; vote for a card; pick cards if all players have voted
        else:
            next_player = self.game.execute_game_loop(action)
            logger.debug(f"Moving to player id: {next_player}")
            self.current_player_num = next_player
            
            # Check end-of-game condition (no donuts less than no of spaces)
            if self.game.is_game_over():
                reward = self.score_game()
                done = True
            else:
                pass #self.render()

        self.done = done

        return self.observation, reward, done, {}

    def reset(self):

        # for testing human play with original deck
        deck_filter = GoNutsGame.teal_deck_filter()
        self.game.reset_game(shuffle=True, deck_filter=deck_filter)
        self.game.start_game()

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
