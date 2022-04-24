
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutDeckPosition, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered
import gym
import numpy as np
import math
import gonutsfordonuts.envs.cards as cards
import gonutsfordonuts.envs.actions as actions

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

            score_rv = GoNutsScorer.score_red_velvet(position)
            logger.debug(f'Red Velvet: {score_rv}')
            player_scores[p] += score_rv

            score_spr = GoNutsScorer.score_sprinkled(position)
            logger.debug(f'Sprinkled: {score_spr}')
            player_scores[p] += score_spr

            score_bc = GoNutsScorer.score_boston_cream(position)
            logger.debug(f'Boston Cream: {score_bc}')
            player_scores[p] += score_bc
        
        logger.debug(f'All without plain (all players): {player_scores}')
        score_plain = GoNutsScorer.score_plain(positions)
        logger.debug(f'Plain (all players): {score_plain}')
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
        card_counter = Counter([ c.type for c in position.cards ])
        dh_count = card_counter[cards.TYPE_DH]
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
        card_counter = Counter([ c.type for c in position.cards ])
        dn_count = card_counter[cards.TYPE_JF]
        if dn_count == 2 or dn_count == 3:
            return 5
        if dn_count == 4 or dn_count == 5:
            return 10

        return 0

    @staticmethod
    def score_glazed(position):
        card_counter = Counter([ c.type for c in position.cards ])
        dn_count = card_counter[cards.TYPE_GZ]
        return dn_count * 2

    @staticmethod
    def score_french_cruller(position):
        card_counter = Counter([ c.type for c in position.cards ])
        dn_count = card_counter[cards.TYPE_FC]
        return dn_count * 2

    @staticmethod
    def score_powdered(position):
        card_counter = Counter([ c.type for c in position.cards ])
        dn_count = card_counter[cards.TYPE_POW]
        return dn_count * 3

    @staticmethod
    def score_red_velvet(position):
        card_counter = Counter([ c.type for c in position.cards ])
        dn_count = card_counter[cards.TYPE_RV]
        return dn_count * -2

    @staticmethod
    def score_sprinkled(position):
        card_counter = Counter([ c.type for c in position.cards ])
        dn_count = card_counter[cards.TYPE_SPR]
        return dn_count * 2

    @staticmethod
    def score_boston_cream(position):
        card_counter = Counter([ c.type for c in position.cards ])
        dn_count = card_counter[cards.TYPE_BC]
        if not dn_count:
            return 0
        bc_scores = [0, 3, 0, 15, 0, 25]
        return bc_scores[dn_count - 1]

    @staticmethod
    def score_maple_bar(position):
        card_counter = Counter([ c.type for c in position.cards ])
        types_of_cards = len(card_counter)
        if types_of_cards > 6:
            return card_counter[cards.TYPE_MB] * 3
        return 0


class GoNutsGameGymTranslator:

    def __init__(self, donuts_game):
        self.game = donuts_game

        # Define the maximum card space for the max numbers of players
        # (for lower number of player games, non-player observations are zeroed)
        self.total_possible_cards = 70
        self.total_possible_card_types = 13
        self.total_possible_players = 3 # Could be up to 5 but keep to 3 to make the model smaller

    def total_positions(self):
        # player positions / discards
        return self.game.n_players + 1

    def observation_space_size(self):

        # player positions / discard / discard top / player scores / legal actions
        # return self.total_possible_card_types * self.total_possible_players + self.total_possible_card_types + self.total_possible_card_types + self.total_possible_players + self.action_space_size()
        # Simplified version
        return self.total_possible_card_types * self.total_possible_players + self.total_possible_card_types + self.total_possible_card_types + self.total_possible_players + self.action_space_size()

    def action_space_size(self):
        # agents choose a card_id as an action.
        # although all card_ids corresponding to the same card type ought to have the same value,
        # it is actual cards we pick and N of them are unmasked each turn
        
        #      pick-style-actions          give-away-style-actions
        return self.total_possible_card_types + self.total_possible_card_types
    
    def get_legal_actions(self, current_player_num):
        
        legal_actions = np.zeros(self.action_space_size())

        if self.game.game_state == GoNutsGameState.PICK_DONUT:
            for i in range(self.game.no_donut_decks):
                legal_actions[self.game.donut_decks[i].card.type] = 1
        elif self.game.game_state == GoNutsGameState.PICK_DISCARD:
            for card in self.game.discard.cards:
                legal_actions[card.type] = 1
        elif self.game.game_state == GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS:
            if self.game.deck.size():
                legal_actions[self.game.deck.peek_one().type] = 1
            if self.game.deck.size() > 1 and self.game.deck.peek_in_nth_position(2):
                legal_actions[self.game.deck.peek_in_nth_position(2).type] = 1
        elif self.game.game_state == GoNutsGameState.GIVE_CARD:
            
            # Sprinkled requires giving a Sprinkled card to only be an option if the player has a sprinkled card already
            if self.game.players[current_player_num].position.contains_id(cards.SPR_FIRST) and self.game.players[current_player_num].position.contains_id(cards.SPR_2):
                legal_actions[actions.ACTION_GIVE_CARD + cards.TYPE_SPR] = 1

            # OR it's the only card in the position
            if self.game.players[current_player_num].position.contains_id(cards.SPR_FIRST) and self.game.players[current_player_num].position.size() == 1:
                legal_actions[actions.ACTION_GIVE_CARD + cards.TYPE_SPR] = 1
            if self.game.players[current_player_num].position.contains_id(cards.SPR_2) and self.game.players[current_player_num].position.size() == 1:
                legal_actions[actions.ACTION_GIVE_CARD + cards.TYPE_SPR] = 1

            # Otherwise it's all cards in the position, excluding Sprinkles cards
            for card in self.game.players[current_player_num].position.cards:
                if not card.id == cards.SPR_FIRST and not card.id == cards.SPR_2:
                    legal_actions[actions.ACTION_GIVE_CARD + card.type] = 1

        else:
            logger.info(f'get_legal_actions called in inappropriate game state {self.game.game_state}')
            raise Exception(f'get_legal_actions called in inappropriate game state {self.game.game_state}')
        
        return legal_actions

    def get_observations(self, current_player_num):

        # To make this a generalisable model
        # Always assume we are playing with 5 players and just 0 out their observations
        # Always assume we are playing with all the cards, cards we are not playing with will not occur

        # NOW MODELLING 3 players and 7 classes for a smaller obvs space

        n_players = self.game.n_players

        # 3 * 13 = 39 obvs

        positions = np.zeros([self.total_possible_players, self.total_possible_card_types])

        # Each player's current position (tableau)
        # starting from the current player and cycling to higher-numbered players
        # use '1' if the player has this TYPE of card and 0 if not

        # create from player 0 perspective
        for i in range(n_players):
            player = self.game.players[i]

            for card in player.position.cards:
               positions[i][card.type] = 1

        positions_flat = positions.flatten()
        # roll forward (+wrap) to put this player's numbers at the start
        positions_rolled = np.roll(positions_flat, (self.total_possible_players - current_player_num) * self.total_possible_card_types)
        ret = positions_rolled

        # 13 obvs (52 so far)

        # The discard deck
        # Again by type
        discard = np.zeros(self.total_possible_card_types)

        for card in self.game.discard.cards:
            discard[card.type] = 1
        
        ret = np.append(ret, discard)

        # 13 obvs (65 so far)

        # The top discard card [for eclair]
        # Currently removed to make the obvs space smaller
        top_discard = np.zeros(self.total_possible_card_types)

        if self.game.discard.size():
            top_discard[self.game.discard.peek_one().type] = 1
        
        ret = np.append(ret, top_discard)

        # Current player scores [to guide the agent to which players to target]
        # 3 obvs (68 so far)
        player_scores = np.zeros(self.total_possible_players)

        for i in range(n_players):
            player = self.game.players[i]
            player_scores[i] = player.score / self.game.max_score

        scores_rolled = np.roll(player_scores, self.total_possible_players - current_player_num)

        ret = np.append(ret, scores_rolled)
        
        # 26 obvs (94 so far)

        # Legal actions, representing the donut choices or other actions
        ret = np.append(ret, self.get_legal_actions(current_player_num))

        return ret

class GoNutsGameState:
    PICK_DONUT = 0
    PICK_DISCARD = 1
    INSTANT_ACTION = 2
    PICK_ONE_FROM_TWO_DECK_CARDS = 3
    GIVE_CARD = 4

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
            self.players.append(Player(player_id))
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
    def teal_deck_filter_no_fc(self):
        return [ cards.CF_FIRST, cards.CF_2, cards.CF_3, cards.DH_FIRST, cards.DH_2, cards.DH_3, cards.DH_4, cards.DH_5, cards.DH_6,
        cards.ECL_FIRST, cards.ECL_2, cards.ECL_3, cards.GZ_FIRST, cards.GZ_2, cards.GZ_3, cards.GZ_4, cards.GZ_5,
        cards.JF_FIRST, cards.JF_2, cards.JF_3, cards.JF_4, cards.JF_5, cards.JF_6, cards.MB_FIRST, cards.MB_2,
        cards.P_FIRST, cards.P_2, cards.P_3, cards.P_4, cards.P_5, cards.P_6, cards.P_7,
        cards.POW_FIRST, cards.POW_2, cards.POW_3, cards.POW_4]

    @classmethod
    def teal_and_pink_filter(self):
        return [ cards.CF_FIRST, cards.CF_2, cards.CF_3, cards.DH_FIRST, cards.DH_2, cards.DH_3, cards.DH_4, cards.DH_5, cards.DH_6,
        cards.ECL_FIRST, cards.ECL_2, cards.ECL_3, cards.GZ_FIRST, cards.GZ_2, cards.GZ_3, cards.GZ_4, cards.GZ_5,
        cards.JF_FIRST, cards.JF_2, cards.JF_3, cards.JF_4, cards.JF_5, cards.JF_6, cards.MB_FIRST, cards.MB_2,
        cards.P_FIRST, cards.P_2, cards.P_3, cards.P_4, cards.P_5, cards.P_6, cards.P_7,
        cards.POW_FIRST, cards.POW_2, cards.POW_3, cards.POW_4, cards.FC_FIRST, cards.FC_2,
        cards.BC_FIRST, cards.BC_2, cards.BC_3, cards.BC_4, cards.BC_5, cards.BC_6,
        cards.DC_FIRST, cards.DC_2, cards.RV_FIRST, cards.RV_2, cards.SPR_FIRST, cards.SPR_2 ]

    # fc is excluded since its card ID is 22 which puts it out-of-range unless using all card types
    @classmethod
    def teal_and_pink_filter_no_fc(self):
        return [ cards.CF_FIRST, cards.CF_2, cards.CF_3, cards.DH_FIRST, cards.DH_2, cards.DH_3, cards.DH_4, cards.DH_5, cards.DH_6,
        cards.ECL_FIRST, cards.ECL_2, cards.ECL_3, cards.GZ_FIRST, cards.GZ_2, cards.GZ_3, cards.GZ_4, cards.GZ_5,
        cards.JF_FIRST, cards.JF_2, cards.JF_3, cards.JF_4, cards.JF_5, cards.JF_6, cards.MB_FIRST, cards.MB_2,
        cards.P_FIRST, cards.P_2, cards.P_3, cards.P_4, cards.P_5, cards.P_6, cards.P_7,
        cards.POW_FIRST, cards.POW_2, cards.POW_3, cards.POW_4,
        cards.BC_FIRST, cards.BC_2, cards.BC_3, cards.BC_4, cards.BC_5, cards.BC_6,
        cards.DC_FIRST, cards.DC_2, cards.RV_FIRST, cards.RV_2, cards.SPR_FIRST, cards.SPR_2 ]

    @classmethod
    def test_pink_filter(self):
        return [ cards.DH_FIRST, cards.DH_2, cards.DH_3, cards.DH_4, cards.DH_5, cards.DH_6,
        cards.GZ_FIRST, cards.GZ_2, cards.GZ_3, cards.GZ_4, cards.GZ_5,
        cards.JF_FIRST, cards.JF_2, cards.JF_3, cards.JF_4, cards.JF_5, cards.JF_6, cards.MB_FIRST, cards.MB_2,
        cards.RV_FIRST, cards.RV_2, cards.DC_FIRST, cards.DC_2, cards.SPR_FIRST, cards.SPR_2 ]


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
    
    def position_card_for_card_id(self, player_no, card_id):
        for card in self.players[player_no].position.cards:
            if card.id == card_id:
                return card

        logger.info(f'Cannot find card_id {card_id} in position of player {player_no}')
        raise Exception(f'Cannot find card_id {card_id} in position of player {player_no}')

    def deck_for_card_id(self, card_id):
        for deck in self.donut_decks:
            if deck.card.id == card_id:
                return deck

        logger.info(f'Cannot find deck for card_id {card_id}')
        raise Exception(f'Cannot find deck for card_id {card_id}')

    def discard_card_for_card_id(self, card_id):
        for card in self.discard.cards:
            if card.id == card_id:
                return card

        logger.info(f'Cannot find card_id {card_id} in discard')
        raise Exception(f'Cannot find card_id {card_id} in discard')

    def deck_card_for_card_id(self, card_id):
        for card in self.deck.cards:
            if card.id == card_id:
                return card

        logger.info(f'Cannot find card_id {card_id} in deck')
        raise Exception(f'Cannot find card_id {card_id} in deck')

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
                logger.info(f'Player {player.id} picks {deck.card.symbol} ({deck.card.type}:{deck.card.id})')

                cards_picked.append(deck.card)
                deck.set_taken()
        
        return cards_picked

    def do_immediate_card_special_effects(self, player_no, card):
        """Do card special effects that don't need a game state change"""

        if card:
            
            if card.type == cards.TYPE_CF:
                self.card_action_chocolate_frosted(player_no)
            elif card.type == cards.TYPE_ECL:
                self.card_action_eclair(player_no)
            else:
                logger.info(f"No instant effect for card {card.symbol} for player {player_no}")


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
        logger.info(f"Card action pick discard [Red Velvet] (draw card from discard) for player {player_no}")

        discard_card_to_pick = self.discard_card_for_card_id(player_discard_pick)

        logger.debug(f"Adding {discard_card_to_pick.symbol} to position of {player_no}")
        self.players[player_no].position.add_one(discard_card_to_pick)
        self.discard.remove_one(discard_card_to_pick)

    def do_pick_one_from_two_deck_action(self, player_no, player_deck_pick):
        logger.info(f"Card action pick from deck [Double Chocolate] (draw one of two cards from deck) for player {player_no}")

        deck_card_to_pick = self.deck_card_for_card_id(player_deck_pick)

        logger.debug(f"Adding {deck_card_to_pick.symbol} to position of {player_no}")
        self.players[player_no].position.add_one(deck_card_to_pick)
        self.deck.remove_one(deck_card_to_pick)

    def do_give_card_action(self, player_no, player_card_to_give):
        logger.info(f"Card action give card [Sprinkled] (give one card from position) for player {player_no}")

        give_card = self.position_card_for_card_id(player_no, player_card_to_give)

        # To simplify in this iteration always give to the player with the lowest score, excluding the current player
        lowest_score = min([ p.score for p in self.players if not p.id == player_no ])
        target_player_no = 0
        for p in self.players:
            if p.score == lowest_score and not p.id == player_no:
                target_player_no = p.id
                break
        
        logger.debug(f"Giving card {give_card.symbol} to position of player {target_player_no}")
        self.players[target_player_no].position.add_one(give_card)
        self.players[player_no].position.remove_one(give_card)

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
            logger.debug(f"Checking special cards for player {self.action_player}, card is {card.name}")
            if card.type == cards.TYPE_RV:
                # skip the state if there are no discard cards
                if self.discard.size() > 0:
                    logger.debug(f"Red velvet change state, sufficient discard ({self.discard.size()}) cards left for action")
                    new_state = GoNutsGameState.PICK_DISCARD
                else:
                    logger.debug(f"Red velvet change state, insufficient discard ({self.discard.size()}) cards left for action")
            if card.type == cards.TYPE_DC:
                # skip the state if there are no deck cards left
                if self.deck.size() > 0:
                    logger.debug(f"Double chocolate change state, sufficient ({self.deck.size()}) cards left for action")
                    new_state = GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS
                else:
                    logger.debug(f"Double chocolate change state, insufficient ({self.deck.size()}) cards left for action")
            if card.type == cards.TYPE_SPR:
                logger.debug(f"Sprinkled, always change state")
                new_state = GoNutsGameState.GIVE_CARD
  
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

            discard_action = self.translate_step_action(self.game_state, self.action_player, step_action)
            self.do_pick_discard_action(self.action_player, discard_action)
            # Move to the next player to have an action
            self.move_to_next_action_player()
            return self.check_action_for_this_action_player_and_set_state()
        
        # Pick one from two deck cards - requires step, action state
        elif self.game_state == GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS:

            deck_action = self.translate_step_action(self.game_state, self.action_player, step_action)
            self.do_pick_one_from_two_deck_action(self.action_player, deck_action)
            # Move to the next player to have an action
            self.move_to_next_action_player()
            return self.check_action_for_this_action_player_and_set_state()

        # Give card from position - requires step, action state
        elif self.game_state == GoNutsGameState.GIVE_CARD:

            give_action = self.translate_step_action(self.game_state, self.action_player, step_action)
            self.do_give_card_action(self.action_player, give_action)
            # Move to the next player to have an action
            self.move_to_next_action_player()
            return self.check_action_for_this_action_player_and_set_state()

        # CHECK DONUT STATES

        # Do player picks card - requires step, donut state
        elif self.game_state == GoNutsGameState.PICK_DONUT:

            donut_action = self.translate_step_action(self.game_state, self.donut_player, step_action)
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
    
    def translate_step_action(self, game_state, current_player_no, step_action):
        if game_state == GoNutsGameState.PICK_DONUT:
            step_action_norm = step_action - actions.ACTION_DONUT
            # action is a card type, so pick a random donut of that type from the available positions
            matching_donut_positions = [ p.card.id for p in self.donut_decks if p.card.type == step_action_norm ]
            if not matching_donut_positions:
                logger.error(f"Can't find donut of type {step_action_norm} in positions")
                raise RuntimeError(f"Can't find donut of type {step_action_norm} in positions")
            card_id_to_choose = random.choice(matching_donut_positions)
            if len(matching_donut_positions) > 1:
                logger.info(f"More than 1 card of type {step_action_norm}, random choice of card id {card_id_to_choose}")
            return card_id_to_choose
        elif game_state == GoNutsGameState.PICK_DISCARD:
            step_action_norm = step_action - actions.ACTION_DISCARD
            # action is a card type, so pick a random donut of that type from the available discard
            matching_discards = [ c.id for c in self.discard.cards if c.type == step_action_norm ]
            if not matching_discards:
                logger.error(f"Can't find donut of type {step_action_norm} in discards")
                raise RuntimeError(f"Can't find donut of type {step_action_norm} in discards")
            return random.choice(matching_discards)
        elif game_state == GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS:
            step_action_norm = step_action - actions.ACTION_DECK
            if self.deck.peek_one().type == step_action_norm:
                return self.deck.peek_one().id
            elif self.deck.peek_in_nth_position(2):
                return self.deck.peek_in_nth_position(2).id
            else:
                logger.error(f"Can't find donut of type {step_action_norm} in 2-from-deck-pick")
                raise RuntimeError(f"Can't find donut of type {step_action_norm} in 2-from-deck-pick")
        elif game_state == GoNutsGameState.GIVE_CARD:
            step_action_norm =  step_action - actions.ACTION_GIVE_CARD
            matching_hand_cards = [ c.id for c in self.players[current_player_no].position.cards if c.type == step_action_norm ]
            if not matching_hand_cards:
                logger.error(f"Can't find donut of type {step_action_norm} in position of player {current_player_no}")
                raise RuntimeError(f"Can't find donut of type {step_action_norm} in position of player {current_player_no}")
            return random.choice(matching_hand_cards)
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
        return self.translator.get_legal_actions(self.current_player_num)

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

        # setup card selection to be used in simulation
        deck_filter = GoNutsGame.teal_deck_filter_no_fc()
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
        print(f"It is Player {self.current_player.id}'s turn to choose in state {self.game.game_state}")            

        # Render player positions

        for p in self.game.players:
            print(f'Player {p.id}\'s position')
            if p.position.size() > 0:
                print('  '.join([card.symbol + ': ' + str(card.id) for card in sorted(p.position.cards, key=lambda x: x.id)]))
            else:
                print('Empty')

        # Render donuts to choose
        for i, d in enumerate(self.game.donut_decks):
            this_card = d.get_card()
            print(f'Deck {i}: {this_card.symbol} ({this_card.type}:{this_card.id})')

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
                    if i > actions.ACTION_GIVE_CARD:
                        real_card_type = i - actions.ACTION_GIVE_CARD
                    else:
                        real_card_type = i
                    # TODO: Have a sensible lookup for type -> card data rather than looking for the first card of this type
                    card_for_action = next((c for c in self.game.deck.base_deck if c.type == real_card_type), None)
                    if card_for_action:
                        legal_action_str += f"{i}:{card_for_action.symbol} "
                    else:
                        print(f"Can't find card for action {i}")
            print(legal_action_str)

        if self.done:
            print(f'\n\nGAME OVER')
            
        for p in self.game.players:
            print(f'Player {p.id} points: {p.score}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Go Nuts For Donuts!')
