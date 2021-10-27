
from app.environments.gonutsfordonuts.gonutsfordonuts.envs.classes import ChocolateFrosted, DonutDeckPosition, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered
import gym
import numpy as np
import math

import config

from stable_baselines import logger

from .classes import *

class GoNutsForDonutsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(GoNutsForDonutsEnv, self).__init__()
        self.name = 'gonutsfordonuts'
        self.manual = manual
        
        self.n_players = 3
        self.no_donut_decks = self.n_players + 1
        #self.cards_per_player = 9
        self.card_types = 9
        
        self.n_rounds = 1
        self.max_score = 200 # to normalise current scores to [0, 1] interval
        
        # player positions / last_choices / discards
        self.total_positions = self.n_players + 2

        self.contents = [
          {'card': ChocolateFrosted, 'info': {}, 'count': 3}  #0 
           ,  {'card': DonutHoles, 'info': {}, 'count':  6} #1 
        ,  {'card': Eclair, 'info': {}, 'count':  3}  #2   
          ,  {'card': Glazed, 'info': {}, 'count':  5} #3  
           ,  {'card': JellyFilled, 'info': {}, 'count':  6} #4 
           ,  {'card': MapleBar,  'info': {}, 'count':  2} #5 
           ,  {'card': Plain, 'info': {}, 'count':  7} #6 
          ,  {'card': Powdered, 'info': {}, 'count':  4}  #7 
          ,  {'card': FrenchCruller, 'info': {}, 'count':  math.min(self.n_players - 1, 4)}  #8 (last due to variability) 
        ]

        self.total_cards = sum([x['count'] for x in self.contents])

        # sushi action space - play any card type (that you have in hand) or any 2 card types (chopsticks)
        # 2 choices for donuts action space - play any card type (that's available in the choices) or play any card_id
        # that's available in the choices; although the latter is degenerate in theory an opponent would always picks
        # the lower-numbered copy of duplicate cards would break this symmetry and make it worth modelling
        # OR we pick a (type, [0..5]) tuple pair
        # Let's go for card_types and model the tuple separately...
        # self.action_space = gym.spaces.Discrete(self.card_types + self.card_types * self.card_types)
        self.action_space = gym.spaces.Discrete(self.card_types)
        # total positions (see above) / player scores / legal actions
        self.observation_space = gym.spaces.Box(0, 1, (self.total_cards * self.total_positions + self.n_players + self.action_space.n ,))
        self.verbose = verbose

        
    @property
    def observation(self):
        # Each player may have each individual card
        obs = np.zeros(([self.total_positions, self.total_cards]))
        player_num = self.current_player_num

        # Note that tidying the observations to card_id strongly couples the model to the exact
        # composition of the original deck. e.g. increasing the number of FCs due to player count may
        # cause the model to be non-generalisable to different player counts
        for i in range(self.n_players):
            player = self.players[player_num]

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
        if self.turns_taken >= 1:
            for card in self.choices_taken:
                obs[self.n_players + 1][card.id] = 1

        # Current player scores [to guide to which players to target]
        ret = obs.flatten()
        for p in self.players: # TODO this should be from reference point of the current_player
            ret = np.append(ret, p.score / self.max_score)

        # Legal actions, representing the donut choices
        ret = np.append(ret, self.legal_actions)

        return ret

    @property
    def legal_actions(self):
        legal_actions = np.zeros(self.action_space.n)

        # Action space - all card ids masked by the 1 to 5 available?
        # Players pick a card id and may duplicate



        for i in range(len(hand)):
            legal_actions[hand[i].order] = 1
            if 'chopsticks' in [x.type for x in self.current_player.position.cards]:
                for j in range(i+1, len(hand)):
                    legal_actions[self.card_types + (hand[i].order * self.card_types) + hand[j].order] = 1
                    legal_actions[self.card_types + (hand[j].order * self.card_types) + hand[i].order] = 1
        
        
        return legal_actions





    def score_game(self):
        reward = [0.0] * self.n_players
        scores = [p.score for p in self.players]
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


    def score_round(self):

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


    @property
    def current_player(self):
        return self.players[self.current_player_num]

    def convert_action(self, action):
        if action < self.card_types:
            return False, action, None
        else:
            action = action - self.card_types
            first_card = action // self.card_types
            second_card = action % self.card_types
            return True, first_card, second_card


    def pickup_chopsticks(self, player):
        logger.debug(f'Player {player.id} picking up chopsticks')
        chopsticks = player.position.pick('chopsticks')
        player.hand.add([chopsticks])

    def play_card(self, card_num, player):

        card_name = self.contents[card_num]['info']['name']
        card = player.hand.pick(card_name)
        if card is None:
            logger.debug(f"Player {player.id} trying to play {card_num} but doesn't exist!")
            raise Exception('Card not found')

        logger.debug(f"Player {player.id} playing {str(card.order) + ': ' + card.symbol + ': ' + str(card.id)}")
        if card.type == 'nigiri':
            for c in player.position.cards:
                if c.type == 'wasabi' and c.played_upon == False:
                    c.played_upon = True
                    card.played_on_wasabi = True
                    break

        player.position.add([card])


    def switch_hands(self):
        logger.debug(f'\nSwitching hands...')
        playernhand = self.players[-1].hand

        for i in range(self.n_players - 1, -1, -1):
            if i > 0:
                self.players[i].hand = self.players[i-1].hand


        self.players[0].hand = playernhand


    def step(self, action):
        
        reward = [0] * self.n_players
        done = False

        # check move legality
        if self.legal_actions[action] == 0:
            reward = [1.0/(self.n_players-1)] * self.n_players
            reward[self.current_player_num] = -1
            done = True

        # pick a card
        else:
            self.action_bank.append(action)

            if len(self.action_bank) == self.n_players:
                logger.debug(f'\nThe chosen cards are now played simultaneously')
                
                for i, action in enumerate(self.action_bank):
                    player = self.players[i]

                    # Look for duplicates

                    # Non-duplicate players play_card()


                    #pickup_chopsticks, first_card, second_card = self.convert_action(action)
                    self.play_card(first_card, player)

                self.action_bank = []
                self.reset_turn()
            
            self.current_player_num = (self.current_player_num + 1) % self.n_players

            if self.current_player_num == 0:
                self.turns_taken += 1

            # Check end-of-game condition (no donuts less than no of spaces)
            if self.remaining_donuts < self.no_of_donuts_to_select:

                reward = self.score_game()
                done = True
            else:
                self.render()
                self.reset_round()

        self.done = done

        return self.observation, reward, done, {}

    def reset_turn(self):
        logger.debug(f'\nResetting turn...')
        
        # Setup donut decks for the first time
        if not self.turns_taken:
            self.donut_decks = []
            for i in range (0, self.no_donut_decks):
                self.donut_decks[i] = DonutDeckPosition(self.deck.draw(1))        
        
        # Redraw and discard any used decks
        for i in range(0, self.no_donut_decks):
            if self.donut_decks[i].taken:
                # Already added to player's position
                logger.debug(f'Filling deck position {i}')
                self.donut_decks[i] = DonutDeckPosition(self.deck.draw(1))
                break
            if self.donut_decks[i].to_discard:
                logger.debug(f'Filling and discarding deck position {i}')
                self.discard.add([self.donut_decks[i].card])
                self.donut_decks[i] = DonutDeckPosition(self.deck.draw(1))
                break
            # Otherwise the card stays in position

    def reset_round(self):

        self.reset_turn()
        self.round += 1
        self.turns_taken = 0

    def reset(self):
        self.round = 0
        self.deck = Deck(self.contents)
        self.discard = Discard()
        self.players = []
        self.action_bank = []

        player_id = 1
        for p in range(self.n_players):
            self.players.append(Player(str(player_id)))
            player_id += 1

        self.current_player_num = 0
        self.done = False
        self.reset_round()
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
        raise Exception('Rules based agent is not yet implemented for Sushi Go!')
