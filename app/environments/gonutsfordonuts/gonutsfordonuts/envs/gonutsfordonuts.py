
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutDeckPosition, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered
import gym
import numpy as np
import math

import config

from stable_baselines import logger
from collections import Counter

from .classes import *

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
            if deck.card_id == card_id:
                return deck

        logger.debug(f'Cannot find deck for card_id {card_id}')
        raise Exception(f'Cannot find deck for card_id {card_id}')

    def pick_cards(self, cards_ids_picked):
        
        if len(cards_ids_picked) != self.n_players:
            logger.debug('pick_cards() called with wrong number of card_ids')
            raise Exception('pick_cards() called with wrong number of card_ids')

        for i, card_id in enumerate(cards_ids_picked):
        
            card_ids_counter = Counter(cards_ids_picked)

            player = self.players[i]
            deck_id = self.deck_for_card_id(card_id)

            if card_ids_counter[card_id] > 1:
                self.donut_decks[deck_id].set_to_discard()
            else:
                player.position.add(self.donut_decks[deck_id].card)
                self.donut_decks[deck_id].set_taken()

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
                    break
                if self.donut_decks[i].to_discard:
                    logger.debug(f'Filling and discarding deck position {i}')
                    self.discard.add([self.donut_decks[i].card])
                    self.donut_decks[i] = DonutDeckPosition(self.deck.draw_one())
                    break
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

    def get_limits(self, counts, type):
        counts = np.array(counts, dtype=np.float)
        if type == 'max':
            type_counts = np.nanmax(counts)
        else:
            type_counts = np.nanmin(counts)

        counts_winners = []

        for i, m in enumerate(counts):
            if m == type_counts:
                counts_winners.append(i)
                
        return counts_winners


    def score_puddings(self):
        logger.debug('\nPudding counts...')

        puddings = []
        for p in self.players:
            puddings.append(len([card for card in p.position.cards if card.type == 'pudding']))
        
        logger.debug(f'Puddings: {puddings}')

        pudding_winners = self.get_limits(puddings, 'max')

        for i in pudding_winners:
            self.players[i].score += 6 // len(pudding_winners)
            logger.debug(f'Player {self.players[i].id} 1st place puddings: {6 // len(pudding_winners)}')
        
        pudding_losers = self.get_limits(puddings, 'min')

        for i in pudding_losers:
            self.players[i].score -= 6 // len(pudding_losers)
            logger.debug(f'Player {self.players[i].id} last place puddings: {-6 // len(pudding_losers)}')



    def score_maki(self, maki):
        logger.debug('\nMaki counts...')
        logger.debug(f'Maki: {maki}')

        maki_winners = self.get_limits(maki, 'max')

        for i in maki_winners:
            self.players[i].score += 6 // len(maki_winners)
            maki[i] = None #mask out the winners
            logger.debug(f'Player {self.players[i].id} 1st place maki: {6 // len(maki_winners)}')
        
        if len(maki_winners) == 1:
            #now get second place as winners are masked with None
            maki_winners = self.get_limits(maki, 'max')

            for i in maki_winners:
                self.players[i].score += 3 // len(maki_winners)
                logger.debug(f'Player {self.players[i].id} 2nd place maki: {3 // len(maki_winners)}')


    


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
            self.action_bank.append(action)

            if len(self.action_bank) == self.n_players:
                logger.debug(f'\nThe chosen cards are now competitively picked')

                self.game.pick_cards(self.action_bank)
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
