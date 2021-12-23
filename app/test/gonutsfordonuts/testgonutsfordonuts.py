import numpy as np

from gonutsfordonuts.envs.gonutsfordonuts import GoNutsGame, GoNutsScorer, GoNutsGameGymTranslator, GoNutsForDonutsEnvUtility
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered, BostonCream, DoubleChocolate, RedVelvet, Sprinkled, BearClaw, CinnamonTwist, Coffee, DayOldDonuts, Milk, OldFashioned, MapleFrosted, MuchoMatcha, RaspberryFrosted, StrawberryGlazed
from gonutsfordonuts.envs.classes import Position, Player, Deck

class TestDeck:

    def test_standard_contents(self):

        d = Deck(2, GoNutsGame.standard_deck_contents())

        assert d.size() == 70

    def test_filter_standard_deck(self):

        d = Deck(2, GoNutsGame.standard_deck_contents())

        desired_filtered_cards = [1, 5, 31, 52]

        d.filter(desired_filtered_cards)

        expected_cards = [1, 5, 31, 52]
        # We are inspecting the deck from bottom to top but cards are drawn in the other way
        expected_cards.reverse()
        expected_filtered_cards = np.array(expected_cards)

        new_card_order = np.array([ c.id for c in d.cards ])

        assert (new_card_order == expected_filtered_cards).all()

    def test_reorder_standard_deck(self):

        d = Deck(2, GoNutsGame.standard_deck_contents())

        desired_ordered_cards = [10, 20, 30, 34]

        d.reorder(desired_ordered_cards)

        expected_order = [10, 20, 30, 34]
        expected_order.extend(list(range(0, 10)))
        expected_order.extend(list(range(11, 20)))
        expected_order.extend(list(range(21, 30)))
        expected_order.extend(list(range(31, 34)))
        expected_order.extend(list(range(35, 70)))

        # We are inspecting the deck from bottom to top but cards are drawn in the other way
        expected_order.reverse()

        expected_order_cards = np.array(expected_order)

        new_card_order = np.array([ c.id for c in d.cards ])

        assert (new_card_order == expected_order_cards).all()

    def test_deal_reordered_standard_deck(self):

        d = Deck(2, GoNutsGame.standard_deck_contents())

        desired_ordered_cards = [CF_FIRST, DH_FIRST, GZ_FIRST, POW_FIRST]

        d.reorder(desired_ordered_cards)
    
        assert d.draw_one().symbol == "CF"
        assert d.draw_one().symbol == "DH"
        assert d.draw_one().symbol == "GZ"
        assert d.draw_one().symbol == "POW"


class TestGoNutsForDonutsEnvUtility:
    def test_rewards_single_winner(self):

        players = [Player(p) for p in range(4)]
        players[0].score = 0.1
        players[1].score = 0.2
        players[2].score = 0.15
        players[3].score = 0.05

        assert (GoNutsForDonutsEnvUtility.score_game_from_players(players) == np.array([ 0, +1, 0, -1])).all()

    def test_rewards_multiple_winners(self):

        players = [Player(p) for p in range(4)]
        players[0].score = 0.1
        players[1].score = 0.2
        players[2].score = 0.2
        players[3].score = 0.05

        assert (GoNutsForDonutsEnvUtility.score_game_from_players(players) == np.array([0, +0.5, +0.5, -1])).all()

    def test_rewards_multiple_losers(self):

        players = [Player(p) for p in range(4)]
        players[0].score = 0.1
        players[1].score = 0.2
        players[2].score = 0.05
        players[3].score = 0.05

        assert (GoNutsForDonutsEnvUtility.score_game_from_players(players) == np.array([0, +1, -0.5, -0.5])).all()

class TestGoNutsForDonutsScorer:

    def test_score_zero_donut_holes(self):
        position = Position()

        assert GoNutsScorer.score_donut_holes(position) == 0

    def test_score_three_donut_holes(self):
        position = Position()
        position.add([DonutHoles(1,1), DonutHoles(2,1), DonutHoles(3,1)])

        assert GoNutsScorer.score_donut_holes(position) == 6

    def test_score_zero_jelly_filled(self):
        position = Position()

        assert GoNutsScorer.score_jelly_filled(position) == 0

    def test_score_three_jelly_filled(self):
        position = Position()
        position.add([JellyFilled(1,1), JellyFilled(2,1), JellyFilled(3,1)])

        assert GoNutsScorer.score_jelly_filled(position) == 5

    def test_score_three_glazed(self):
        position = Position()
        position.add([Glazed(1,1), Glazed(2,1), Glazed(3,1)])

        assert GoNutsScorer.score_glazed(position) == 6

    def test_score_two_french_cruller(self):
        position = Position()
        position.add([FrenchCruller(1,1), FrenchCruller(2,1)])

        assert GoNutsScorer.score_french_cruller(position) == 4
    
    def test_score_two_powdered(self):
        position = Position()
        position.add([Powdered(1,1), Powdered(2,1)])

        assert GoNutsScorer.score_powdered(position) == 6 

    def test_score_maple_bar_with_six_types(self):
        position = Position()
        position.add([Powdered(1,1), Glazed(2,1), Eclair(3,1), ChocolateFrosted(4,1), ChocolateFrosted(5,1),
        DonutHoles(6,1), MapleBar(7,1)])

        assert GoNutsScorer.score_maple_bar(position) == 0
    
    def test_score_maple_bar_with_seven_types(self):
        position = Position()
        position.add([Powdered(1,1), Glazed(2,1), Eclair(3,1), ChocolateFrosted(4,1), ChocolateFrosted(5,1),
        DonutHoles(6,1), FrenchCruller(7,1), MapleBar(8,1)])

        assert GoNutsScorer.score_maple_bar(position) == 3

    def test_score_two_maple_bars_with_seven_types(self):
        position = Position()
        position.add([Powdered(1,1), Glazed(2,1), Eclair(3,1), ChocolateFrosted(4,1), ChocolateFrosted(5,1),
        DonutHoles(6,1), FrenchCruller(7,1), MapleBar(8,1), MapleBar(9,1)])

        assert GoNutsScorer.score_maple_bar(position) == 6
    
    def test_score_zero_maple_bars_with_seven_types(self):
        position = Position()
        position.add([Powdered(1,1), Glazed(2,1), Eclair(3,1), ChocolateFrosted(4,1), ChocolateFrosted(5,1),
        DonutHoles(6,1), FrenchCruller(7,1), JellyFilled(8,1)])

        assert GoNutsScorer.score_maple_bar(position) == 0

    def test_score_plain_no_winners(self):
        positions = [ Position(), Position(), Position()]
        positions[0].add([Glazed(2,1)])
        positions[1].add([])
        positions[2].add([Glazed(5,1), Glazed(6,1)])

        scores = GoNutsScorer.score_plain(positions)
        assert (scores == [0, 0, 0]).all()

    def test_score_plain_one_winner(self):
        positions = [ Position(), Position(), Position()]
        positions[0].add([Plain(1,1), Glazed(2,1)])
        positions[1].add([Plain(3,1), Plain(4,1)])
        positions[2].add([Glazed(5,1), Glazed(6,1), Plain(7,1)])

        scores = GoNutsScorer.score_plain(positions)
        assert (scores == [1, 5, 1]).all()

    def test_score_plain_tied_winners(self):
        positions = [ Position(), Position(), Position()]
        positions[0].add([Plain(1,1), Glazed(2,1), Plain(7,1)])
        positions[1].add([Plain(3,1), Plain(4,1)])
        positions[2].add([Glazed(5,1), Glazed(6,1)])

        scores = GoNutsScorer.score_plain(positions)
        assert (scores == [3, 3, 0]).all()

    def test_turn_scoring(self):
        positions = [ Position(), Position(), Position()]
        positions[0].add([Plain(1,1), Glazed(2,1), DonutHoles(3,1), DonutHoles(4,1), Plain(16,1)])
        positions[1].add([Powdered(5,1), JellyFilled(6,1), FrenchCruller(7,1), Eclair(8,1), MapleBar(16,1)])
        positions[2].add([Glazed(9,1), ChocolateFrosted(10,1), MapleBar(11,1), JellyFilled(12,1), DonutHoles(13,1), Powdered(14,1), Plain(15,1)])

        scores = GoNutsScorer.score_turn(positions)
        assert (scores == [10, 5, 10]).all()

CF_FIRST = 0
DH_FIRST = 3
ECL_FIRST = 9
GZ_FIRST = 12
JF_FIRST = 17
MB_FIRST = 23
P_FIRST = 25
POW_FIRST = 32
BC_FIRST = 36
DC_FIRST = 42
RV_FIRST = 44
SPR_FIRST = 46
BEAR_FIRST = 48
CT_FIRST = 50
CFF_FIRST = 52
DOD_FIRST = 54
MILK_FIRST = 55
OLD_FIRST = 56
MF_FIRST = 58
MM_FIRST = 60
RF_FIRST = 62
SG_FIRST = 64
FC_FIRST = 66

OBVS_POSITIONS_START = 0
OBVS_ALL_DISCARD_START = 350
OBVS_TOP_DISCARD_START = 420
OBVS_SCORES_START = 490
OBVS_LEGAL_ACTIONS_START = 495

class TestGoNutsForDonutsGymTranslator:

    def fixture_card_order(self):

        #  [ {'card': ChocolateFrosted, 'info': {}, 'count': 3}  #0 
        #    ,  {'card': DonutHoles, 'info': {}, 'count':  6} #1 
        #    ,  {'card': Eclair, 'info': {}, 'count':  3}  #2   
        #    ,  {'card': Glazed, 'info': {}, 'count':  5} #3  
        #    ,  {'card': JellyFilled, 'info': {}, 'count':  6} #4 
        #    ,  {'card': MapleBar,  'info': {}, 'count':  2} #5 
        #    ,  {'card': Plain, 'info': {}, 'count':  7} #6 
        #    ,  {'card': Powdered, 'info': {}, 'count':  4}  #7         
        #    ,  {'card': BostonCream, 'info': {}, 'count':  6} #8
        #    ,  {'card': DoubleChocolate, 'info': {}, 'count':  2} #9
        #    ,  {'card': RedVelvet, 'info': {}, 'count':  2} #10
        #    ,  {'card': Sprinkled, 'info': {}, 'count':  2} #11
        #    ,  {'card': BearClaw, 'info': {}, 'count':  2} #12
        #    ,  {'card': CinnamonTwist, 'info': {}, 'count':  2} #13
        #    ,  {'card': Coffee, 'info': {}, 'count':  2} #14
        #    ,  {'card': DayOldDonuts, 'info': {}, 'count':  1} #15
        #    ,  {'card': Milk, 'info': {}, 'count':  1} #16
        #    ,  {'card': OldFashioned, 'info': {}, 'count':  2} #17
        #    ,  {'card': MapleFrosted, 'info': {}, 'count':  2} #18
        #    ,  {'card': MuchoMatcha, 'info': {}, 'count':  2} #19
        #    ,  {'card': RaspberryFrosted, 'info': {}, 'count':  2} #19
        #    ,  {'card': StrawberryGlazed, 'info': {}, 'count':  2} #20
        #    ,  {'card': FrenchCruller, 'info': {}, 'count':  4}  #21 (last due to variability) 
        # ].reverse()

        #       CF  DH ECL GLZ JF  MB  P   PWD FC
        return [ CF_FIRST, DH_FIRST, ECL_FIRST, GZ_FIRST, JF_FIRST, MB_FIRST, P_FIRST, POW_FIRST, FC_FIRST ]

    def test_legal_actions_have_only_donut_deck_picks(self):

        test_game = GoNutsGame(5)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        # 6 dealt cards will be the top 6
        expected_legal_actions = np.zeros(translator.total_possible_cards)
        for i in self.fixture_card_order()[:6]:
            expected_legal_actions[i] = 1

        assert (translator.get_legal_actions() == expected_legal_actions).all()

    def test_observations_correct_for_started_game(self):

        no_players = 4
        test_game = GoNutsGame(no_players)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        # Positions
        obs = np.zeros([translator.total_possible_players, translator.total_possible_cards])
        ret = obs.flatten()

        # Discard
        ret = np.append(ret, np.zeros(translator.total_possible_cards))
        ret = np.append(ret, np.zeros(translator.total_possible_cards))

        # Scores
        ret = np.append(ret, np.zeros(translator.total_possible_players))

        # Legal actions
        legal_actions = np.zeros(translator.total_possible_cards)
        for i in self.fixture_card_order()[:5]:
            legal_actions[i] = 1
        ret = np.append(ret, legal_actions)

        expected_observations = ret
        
        assert (translator.get_observations(0) == expected_observations).all()

    def observation_comparer(self, actual, expected):

        assert (actual[OBVS_POSITIONS_START:OBVS_ALL_DISCARD_START] == expected[OBVS_POSITIONS_START:OBVS_ALL_DISCARD_START]).all()
        assert (actual[OBVS_ALL_DISCARD_START:OBVS_TOP_DISCARD_START] == expected[OBVS_ALL_DISCARD_START:OBVS_TOP_DISCARD_START]).all()
        assert (actual[OBVS_TOP_DISCARD_START:OBVS_SCORES_START] == expected[OBVS_TOP_DISCARD_START:OBVS_SCORES_START]).all()
        assert (actual[OBVS_SCORES_START:OBVS_LEGAL_ACTIONS_START] == expected[OBVS_SCORES_START:OBVS_LEGAL_ACTIONS_START]).all()
        assert (actual[OBVS_LEGAL_ACTIONS_START:] == expected[OBVS_LEGAL_ACTIONS_START:]).all()
        assert (actual == expected).all()

    def test_observations_correct_for_game_after_one_turn_from_player_index_zero_perspective(self):

        no_players = 4
        test_game = GoNutsGame(no_players)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game() # deals top 5
        # d1: 0 CF, d2: 3 DH, d3: 9 ECL, d4: 12 G, d5: 17 JF; draw: [23 MB, 25 P, 32 PWD, 66 FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_action(ACTION_DONUT, [CF_FIRST, DH_FIRST, DH_FIRST, GZ_FIRST]))
        # p1: [0 CF, 23 MB], p2: [], p3: [], p4: [12 G]; draw: [25 P, 32 PWD, 66 FC]
        test_game.reset_turn()
        # d1: 25 P, d2: 32 PWD, d3: 9 ECL, d4: 66 FC, d5: 17 JF; draw: []; discard: [3 DH]

        # Positions (from player 0 perspective)
        no_cards = translator.total_possible_cards
        obs = np.array([])
        # p0
        p0position = np.zeros(no_cards)
        p0position[CF_FIRST] = 1
        p0position[MB_FIRST] = 1
        obs = np.append(obs, p0position)
        # p1
        obs = np.append(obs, np.zeros(no_cards))
        # p2
        obs = np.append(obs, np.zeros(no_cards))
        # p3
        p3position = np.zeros(no_cards)
        p3position[GZ_FIRST] = 1
        obs = np.append(obs, p3position)
        # p4 (not in the game but included in the obvs space)
        obs = np.append(obs, np.zeros(no_cards))
        ret = obs

        # Discard
        discard = np.zeros(translator.total_possible_cards)
        discard[DH_FIRST] = 1
        ret = np.append(ret, discard)
        ret = np.append(ret, discard)

        # Scores
        ret = np.append(ret, np.array([0, 0, 0, 2 / test_game.max_score, 0]))
        
        # Legal actions
        legal_actions = np.zeros(no_cards)
        legal_actions[P_FIRST] = 1
        legal_actions[POW_FIRST] = 1
        legal_actions[ECL_FIRST] = 1
        legal_actions[FC_FIRST] = 1
        legal_actions[JF_FIRST] = 1
        ret = np.append(ret, legal_actions)

        expected_observations = ret
        
        self.observation_comparer(translator.get_observations(0), expected_observations)

    def test_observations_correct_for_multiple_discards(self):

        no_players = 4
        test_game = GoNutsGame(no_players)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game() # deals top 5
        # d1: 0 CF, d2: 3 DH, d3: 9 ECL, d4: 12 G, d5: 17 JF; draw: [23 MB, 25 P, 32 PWD, 66 FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_action(ACTION_DONUT, [CF_FIRST, DH_FIRST, DH_FIRST, CF_FIRST]))
        # p1: [], p2: [], p3: [], p4: []; draw: [23 MB, 25 P, 32 PWD, 66 FC]
        test_game.reset_turn()
        # d1: 23 MB, d2: 25 P, d3: 9 ECL, d4: 12 G, d5: 17 JF; draw: [32 PWD, 66 FC]; discard: [0 CF, 3 DH]

        # Positions (from player 0 perspective)
        no_cards = translator.total_possible_cards
        obs = np.array([])
        # p0  
        obs = np.append(obs, np.zeros(no_cards))
        # p1
        obs = np.append(obs, np.zeros(no_cards))
        # p2
        obs = np.append(obs, np.zeros(no_cards))
        # p3
        obs = np.append(obs, np.zeros(no_cards))
        # p4 (not in the game but included in the obvs space)
        obs = np.append(obs, np.zeros(no_cards))
        ret = obs

        # Discard
        # All
        discard = np.zeros(translator.total_possible_cards)
        discard[DH_FIRST] = 1
        discard[CF_FIRST] = 1
        ret = np.append(ret, discard)
        # Top
        discard = np.zeros(translator.total_possible_cards)
        discard[DH_FIRST] = 1
        ret = np.append(ret, discard)

        # Scores
        ret = np.append(ret, np.array([0, 0, 0, 0, 0]))
        
        # Legal actions
        legal_actions = np.zeros(no_cards)
        legal_actions[MB_FIRST] = 1
        legal_actions[P_FIRST] = 1
        legal_actions[ECL_FIRST] = 1
        legal_actions[GZ_FIRST] = 1
        legal_actions[JF_FIRST] = 1
        ret = np.append(ret, legal_actions)

        expected_observations = ret

        self.observation_comparer(translator.get_observations(0), expected_observations)

    def test_observations_correct_for_game_after_one_turn_from_player_index_three_perspective(self):

        no_players = 4
        test_game = GoNutsGame(no_players)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game() # deals top 5
        # d1: 0 CF, d2: 3 DH, d3: 9 ECL, d4: 12 G, d5: 17 JF; draw: [23 MB, 25 P, 32 PWD, 66 FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_action(ACTION_DONUT, [CF_FIRST, DH_FIRST, DH_FIRST, GZ_FIRST]))
        # p1: [0 CF, 23 MB], p2: [], p3: [], p4: [12 G]; draw: [25 P, 32 PWD, 66 FC]
        test_game.reset_turn()
        # d1: 25 P, d2: 32 PWD, d3: 9 ECL, d4: 66 FC, d5: 17 JF; draw: []; discard: [3 DH]

        # Positions (from player 3 perspective)
        no_cards = translator.total_possible_cards
        obs = np.array([])
        # p3
        p3position = np.zeros(no_cards)
        p3position[GZ_FIRST] = 1
        obs = np.append(obs, p3position)
        # p4 (not in the game but included in the obvs space)
        obs = np.append(obs, np.zeros(no_cards))
        # p0
        p0position = np.zeros(no_cards)
        p0position[CF_FIRST] = 1
        p0position[MB_FIRST] = 1
        obs = np.append(obs, p0position)
        # p1
        obs = np.append(obs, np.zeros(no_cards))
        # p2
        obs = np.append(obs, np.zeros(no_cards))
        
        ret = obs

        # Discard
        discard = np.zeros(translator.total_possible_cards)
        discard[DH_FIRST] = 1
        ret = np.append(ret, discard)
        ret = np.append(ret, discard)

        # Scores
        ret = np.append(ret, np.array([2 / test_game.max_score, 0, 0, 0, 0]))
        
        # Legal actions
        legal_actions = np.zeros(no_cards)
        legal_actions[P_FIRST] = 1
        legal_actions[POW_FIRST] = 1
        legal_actions[ECL_FIRST] = 1
        legal_actions[FC_FIRST] = 1
        legal_actions[JF_FIRST] = 1
        ret = np.append(ret, legal_actions)

        expected_observations = ret
        
        self.observation_comparer(translator.get_observations(3), expected_observations)

ACTION_DONUT = 0

class TestHelpers:

    @staticmethod
    def step_action(action_type, action_ids):
        
        return [ACTION_DONUT + id for id in action_ids]

class TestGoNutsForDonuts:
    
    def fixture_card_filter(self):
        #       CF  DH ECL GLZ JF  MB  P   PWD FC
        return [ CF_FIRST, DH_FIRST, ECL_FIRST, GZ_FIRST, JF_FIRST, MB_FIRST, P_FIRST, POW_FIRST, FC_FIRST ]

    def fixture_card_order(self):
        #       CF  DH ECL GLZ JF  MB  P   PWD FC
        return [ CF_FIRST, DH_FIRST, ECL_FIRST, GZ_FIRST, JF_FIRST, MB_FIRST, P_FIRST, POW_FIRST, FC_FIRST ]

    def test_standard_deck_contents_size(self):

        test_game = GoNutsGame(2)
        test_game.setup_game()
        assert test_game.deck.size() == 70

    def test_override_order_deck_contents_size(self):

        test_game = GoNutsGame(2)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        assert test_game.deck.size() == 9

    def test_decks_setup_for_first_turn(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        assert test_game.donut_decks[0].card.id == CF_FIRST
        assert test_game.donut_decks[0].card.symbol == 'CF'

        assert test_game.donut_decks[1].card.id == DH_FIRST
        assert test_game.donut_decks[1].card.symbol == 'DH'

        assert test_game.donut_decks[2].card.id == ECL_FIRST
        assert test_game.donut_decks[2].card.symbol == 'ECL'

        assert test_game.donut_decks[3].card.id == GZ_FIRST
        assert test_game.donut_decks[3].card.symbol == 'GZ'

        assert test_game.donut_decks[4].card.id == JF_FIRST
        assert test_game.donut_decks[4].card.symbol == 'JF'
    
    def test_uncontested_picks_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        test_game.pick_cards([CF_FIRST, DH_FIRST, ECL_FIRST, GZ_FIRST])

        assert test_game.players[0].position.cards[0].id == CF_FIRST
        assert test_game.players[1].position.cards[0].id == DH_FIRST
        assert test_game.players[2].position.cards[0].id == ECL_FIRST
        assert test_game.players[3].position.cards[0].id == GZ_FIRST

    def test_2nd_round_picks_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        test_game.pick_cards([CF_FIRST, DH_FIRST, ECL_FIRST, GZ_FIRST])
        test_game.reset_turn()

        test_game.pick_cards([JF_FIRST, MB_FIRST, P_FIRST, POW_FIRST])

        assert test_game.players[0].position.cards[1].id == JF_FIRST
        assert test_game.players[1].position.cards[1].id == MB_FIRST
        assert test_game.players[2].position.cards[1].id == P_FIRST
        assert test_game.players[3].position.cards[1].id == POW_FIRST

    def test_game_ends_when_not_enough_donuts_to_refill_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        # 5 out, 4 in deck
        test_game.pick_cards([CF_FIRST, CF_FIRST, DH_FIRST, DH_FIRST])
        test_game.reset_turn()
        # 2 removed, 5 out, 2 in deck
        test_game.pick_cards([ECL_FIRST, GZ_FIRST, JF_FIRST, JF_FIRST])
        test_game.reset_turn()
        # 3 removed, 5 out, -1 in deck

        assert test_game.is_game_over() == True

    def test_game_is_not_over_when_donuts_refilled_but_zero_draw_left(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        # 5 out, 4 in deck
        test_game.pick_cards([CF_FIRST, CF_FIRST, DH_FIRST, DH_FIRST])
        test_game.reset_turn()
        # 2 removed, 5 out, 2 in deck
        test_game.pick_cards([ECL_FIRST, JF_FIRST, JF_FIRST, JF_FIRST])
        test_game.reset_turn()
        # 2 removed, 5 out, 0 in deck

        assert test_game.is_game_over() == False

    def test_empty_decks_get_refilled(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        test_game.pick_cards([CF_FIRST, DH_FIRST, ECL_FIRST, GZ_FIRST])
        test_game.reset_turn()

        assert test_game.donut_decks[0].card.id == MB_FIRST
        assert test_game.donut_decks[0].card.symbol == 'MB'

        assert test_game.donut_decks[1].card.id == P_FIRST
        assert test_game.donut_decks[1].card.symbol == 'P'

        assert test_game.donut_decks[2].card.id == POW_FIRST
        assert test_game.donut_decks[2].card.symbol == 'POW'

        assert test_game.donut_decks[3].card.id == FC_FIRST
        assert test_game.donut_decks[3].card.symbol == 'FC'

    def test_contested_picks_dont_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        test_game.pick_cards([CF_FIRST, CF_FIRST, ECL_FIRST, ECL_FIRST])

        assert len(test_game.players[0].position.cards) == 0
        assert len(test_game.players[1].position.cards) == 0
        assert len(test_game.players[2].position.cards) == 0
        assert len(test_game.players[3].position.cards) == 0

    def test_cards_picked_or_None_returned_by_pick_cards(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        cards_picked = test_game.pick_cards([CF_FIRST, CF_FIRST, ECL_FIRST, GZ_FIRST])

        assert cards_picked[0] == None
        assert cards_picked[1] == None
        assert cards_picked[2].id == ECL_FIRST
        assert cards_picked[3].id == GZ_FIRST

    def test_empty_decks_get_refilled_partial_picks(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        test_game.pick_cards([CF_FIRST, CF_FIRST, ECL_FIRST, GZ_FIRST])
        test_game.reset_turn()

        # Deck 1 - 8 - discarded and refilled
        assert test_game.donut_decks[0].card.id == MB_FIRST
        assert test_game.donut_decks[0].card.symbol == 'MB'

        # Deck 2 - 7 - retained
        assert test_game.donut_decks[1].card.id == DH_FIRST
        assert test_game.donut_decks[1].card.symbol == 'DH'

        # Deck 3 - 6 - taken and refilled
        assert test_game.donut_decks[2].card.id == P_FIRST
        assert test_game.donut_decks[2].card.symbol == 'P'

        # Deck 4 - 5 - taken and refilled
        assert test_game.donut_decks[3].card.id == POW_FIRST
        assert test_game.donut_decks[3].card.symbol == 'POW'

    def test_discard_pile_populated_with_contested_picks(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())

        test_game.start_game()

        test_game.pick_cards([CF_FIRST, CF_FIRST, DH_FIRST, DH_FIRST])
        test_game.reset_turn()

        assert len(test_game.discard.cards) == 2
        assert test_game.discard.cards[0].id == CF_FIRST
        assert test_game.discard.cards[1].id == DH_FIRST
        assert test_game.discard.peek_one().id == DH_FIRST

    def test_card_action_chocolate_frosted_draws_one_from_deck(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game() # deals top 5

        test_game.card_action_chocolate_frosted(player_no=0)

        # first from top of deck is MB_FIRST
        assert test_game.players[0].position.cards[0].id == MB_FIRST
        # 3 cards left in deck
        assert test_game.deck.size() == 3

    def test_card_action_chocolate_frosted_draws_none_from_empty_deck(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()
        # deck 9, deal 5, redeal 4, empty deck
        test_game.pick_cards([CF_FIRST, DH_FIRST, ECL_FIRST, GZ_FIRST])
        test_game.reset_turn()

        test_game.card_action_chocolate_frosted(0)

        # Just the pick, not the additional CF action
        assert test_game.players[0].position.size() == 1

    def test_card_action_eclair_draws_one_from_discard_pile(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game() # deals top 5
        test_game.pick_cards([CF_FIRST, CF_FIRST, DH_FIRST, DH_FIRST]) # adds 8 then 7 to discard pile
        test_game.reset_turn()

        assert len(test_game.discard.cards) == 2

        test_game.card_action_eclair(0)

        # first from top of discard pile is card DH
        assert test_game.players[0].position.cards[0].id == DH_FIRST
        assert len(test_game.discard.cards) == 1
        assert test_game.discard.cards[0].id == CF_FIRST

    def test_card_action_eclair_draws_none_from_empty_discard_pile(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game() # deals top 5

        test_game.card_action_eclair(0)

        assert test_game.players[0].position.size() == 0

    def test_full_game_scoring(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game() # deals top 5
        # d1: 8 CF, d2: 7 DH, d3: 6 ECL, d4: 5 G, d5: 4 JF; draw: [3 MB, 2 P, 1 PWD, 0 FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_action(ACTION_DONUT, [CF_FIRST, DH_FIRST, DH_FIRST, GZ_FIRST]))
        # p1: [8 CF, 3 MB], p2: [], p3: [], p4: [5 G]; draw: [2 P, 1 PWD, 0 FC];
        test_game.reset_turn()
        # d1: 2 P, d2: 1 PWD, d3: 6 ECL, d4: 0 FC, d5: 4 JF; draw: []; discard: [7 DH]
        test_game.execute_game_loop_with_actions(TestHelpers.step_action(ACTION_DONUT, [JF_FIRST, P_FIRST, POW_FIRST, ECL_FIRST]))
        # p1: [8 CF, 3 MB, 4 JF], p2: [2 P], p3: [1 PWD], p4: [5 G, 6 ECL, 7 DH]; draw: []; discard: []

        assert test_game.is_game_over() == True
        assert test_game.player_scores() == [ 0, 4, 3, 3 ]

        test_game.reset_turn()
    



        
    