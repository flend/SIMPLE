import numpy as np

from gonutsfordonuts.envs.gonutsfordonuts import GoNutsGame, GoNutsScorer, GoNutsGameGymTranslator, GoNutsForDonutsEnvUtility, GoNutsGameState
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered, BostonCream, DoubleChocolate, RedVelvet, Sprinkled, BearClaw, CinnamonTwist, Coffee, DayOldDonuts, Milk, OldFashioned, MapleFrosted, MuchoMatcha, RaspberryFrosted, StrawberryGlazed
from gonutsfordonuts.envs.classes import Position, Player, Deck, Discard
import gonutsfordonuts.envs.cards as cards
import gonutsfordonuts.envs.obvs as obvs
import gonutsfordonuts.envs.actions as actions

from stable_baselines import logger

logger.set_level(10)

class TestDiscard:
    def test_remove_contained_card_from_discard(self):
        d = Discard()
        list_of_cards = [Powdered(1,1), Glazed(2,1), Eclair(3,1), ChocolateFrosted(4,1), ChocolateFrosted(5,1),
        DonutHoles(6,1), FrenchCruller(7,1), MapleBar(8,1), MapleBar(9,1)]
        d.add(list_of_cards)

        assert d.size() == 9
        d.remove_one(list_of_cards[1])
        assert d.size() == 8

    def test_remove_non_contained_card_from_discard_by_equality(self):
        d = Discard()
        list_of_cards = [Powdered(1,1), Glazed(2,1), Eclair(3,1), ChocolateFrosted(4,1), ChocolateFrosted(5,1),
        DonutHoles(6,1), FrenchCruller(7,1), MapleBar(8,1), MapleBar(9,1)]
        d.add(list_of_cards)

        assert d.size() == 9
        d.remove_one(Glazed(2,1))
        assert d.size() == 8


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

        desired_ordered_cards = [cards.CF_FIRST, cards.DH_FIRST, cards.GZ_FIRST, cards.POW_FIRST]

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

    def test_score_one_boston_cream(self):
        position = Position()
        position.add([BostonCream(1,1)])

        assert GoNutsScorer.score_boston_cream(position) == 0

    def test_score_two_boston_creams(self):
        position = Position()
        position.add([BostonCream(1,1), BostonCream(2,1)])

        assert GoNutsScorer.score_boston_cream(position) == 3

    def test_score_three_boston_creams(self):
        position = Position()
        position.add([BostonCream(1,1), BostonCream(2,1), BostonCream(3,1)])

        assert GoNutsScorer.score_boston_cream(position) == 0

    def test_score_four_boston_creams(self):
        position = Position()
        position.add([BostonCream(1,1), BostonCream(2,1), BostonCream(3,1), BostonCream(4,1)])

        assert GoNutsScorer.score_boston_cream(position) == 15

    def test_score_five_boston_creams(self):
        position = Position()
        position.add([BostonCream(1,1), BostonCream(2,1), BostonCream(3,1), BostonCream(4,1), BostonCream(5,1)])

        assert GoNutsScorer.score_boston_cream(position) == 0

    def test_score_six_boston_creams(self):
        position = Position()
        position.add([BostonCream(1,1), BostonCream(2,1), BostonCream(3,1), BostonCream(4,1), BostonCream(5,1), BostonCream(6,1)])

        assert GoNutsScorer.score_boston_cream(position) == 25

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

    def test_score_two_red_velvet(self):
        position = Position()
        position.add([RedVelvet(1,1), RedVelvet(2,1)])

        assert GoNutsScorer.score_red_velvet(position) == -4

    def test_score_two_sprinkled(self):
        position = Position()
        position.add([Sprinkled(1,1), Sprinkled(2,1)])

        assert GoNutsScorer.score_sprinkled(position) == 4

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

        #       CF  DH ECL GLZ JF  MB  P   POW FC
        return [ cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.MB_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

    def fixture_card_filter_for_double_chocolate_two_picks(self):
        return [ cards.GZ_FIRST, cards.DC_FIRST, cards.ECL_FIRST, cards.JF_FIRST, cards.MB_FIRST, cards.P_FIRST ]

    def fixture_card_filter_for_double_chocolate_one_pick(self):
        return [ cards.GZ_FIRST, cards.DC_FIRST, cards.ECL_FIRST, cards.JF_FIRST, cards.FC_FIRST ]

    def fixture_card_filter_for_double_chocolate_no_picks(self):
        return [ cards.GZ_FIRST, cards.DC_FIRST, cards.ECL_FIRST, cards.JF_FIRST ]

    def fixture_card_filter_for_sprinkled(self):
        return [ cards.CF_FIRST, cards.DH_FIRST, cards.SPR_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.MB_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

    def fixture_card_filter_for_two_sprinkled(self):
        return [ cards.CF_FIRST, cards.DH_FIRST, cards.SPR_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.SPR_2, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

    def test_observations_is_expected_length(self):

        test_game = GoNutsGame(5)
        translator = GoNutsGameGymTranslator(test_game)
        assert translator.observation_space_size() == 635

    def test_legal_actions_is_expected_length(self):

        test_game = GoNutsGame(4)
        translator = GoNutsGameGymTranslator(test_game)
        assert translator.action_space_size() == 140

    def test_legal_actions_have_only_donut_deck_picks_in_donut_state(self):

        test_game = GoNutsGame(5)
        translator = GoNutsGameGymTranslator(test_game)
        
        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        # 6 dealt cards will be the top 6
        expected_legal_actions = np.zeros(translator.action_space_size())
        for i in self.fixture_card_order()[:6]:
            expected_legal_actions[i] = 1

        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_have_only_discards_picks_in_discard_state(self):

        test_game = GoNutsGame(4)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST]) # adds CF_FIRST then DH_FIRST to discard pile
        test_game.reset_turn()

        test_game.game_state = GoNutsGameState.PICK_DISCARD

        # legal actions are the two discarded cards
        expected_legal_actions = np.zeros(translator.action_space_size())
        expected_legal_actions[cards.CF_FIRST] = 1
        expected_legal_actions[cards.DH_FIRST] = 1
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_have_no_picks_when_no_discard_deck_in_discard_state(self):

        test_game = GoNutsGame(4)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        test_game.game_state = GoNutsGameState.PICK_DISCARD

        # no legal aations
        expected_legal_actions = np.zeros(translator.action_space_size())
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_have_only_two_deck_picks_in_pick_one_of_two_state(self):

        test_game = GoNutsGame(3)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_for_double_chocolate_two_picks())
        test_game.start_game()

        test_game.game_state = GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS

        # legal actions are the two top deck cards
        expected_legal_actions = np.zeros(translator.action_space_size())
        expected_legal_actions[cards.MB_FIRST] = 1
        expected_legal_actions[cards.P_FIRST] = 1
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_have_only_one_deck_picks_in_pick_one_of_two_state_with_one_remaining_card(self):

        test_game = GoNutsGame(3)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_for_double_chocolate_one_pick())
        test_game.start_game()

        test_game.game_state = GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS

        # legal actions are the one remaining deck card
        expected_legal_actions = np.zeros(translator.action_space_size())
        expected_legal_actions[cards.FC_FIRST] = 1
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_are_empty_in_pick_one_of_two_state_with_zero_remaining_cards(self):

        test_game = GoNutsGame(3)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_for_double_chocolate_no_picks())
        test_game.start_game()

        test_game.game_state = GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS

        # no legal actions since no cards left in deck
        expected_legal_actions = np.zeros(translator.action_space_size())
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_include_all_position_cards_except_last_picked_sprinkled_in_give_card_state(self):

        test_game = GoNutsGame(3)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_for_sprinkled())
        test_game.start_game()

        # d1: CF, d2: DH, d3: SPR, d4: GZ; draw: [JF, MB, P, POW, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.DH_FIRST, cards.GZ_FIRST]))
        # p1: [CF, JF], p2: [DH], p3: [GZ]; draw: [MB, P, POW, FC]
        # d1: MB, d2: P, d3: SPR, d4: POW; draw: [FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.SPR_FIRST, cards.P_FIRST, cards.POW_FIRST]))

        test_game.game_state = GoNutsGameState.GIVE_CARD

        # legal actions are position minus the SPR card we just got
        expected_legal_actions = np.zeros(translator.action_space_size())
        expected_legal_actions[TestHelpers.step_action(actions.ACTION_GIVE_CARD, cards.CF_FIRST)] = 1
        expected_legal_actions[TestHelpers.step_action(actions.ACTION_GIVE_CARD, cards.JF_FIRST)] = 1
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_include_all_position_cards_except_one_sprinkled_when_we_have_two_sprinkled_in_give_card_state(self):

        test_game = GoNutsGame(3)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_for_two_sprinkled())
        test_game.start_game()

        # d1: CF, d2: DH, d3: SPR, d4: GZ; draw: [JF, SPR_2, P, POW, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.GZ_FIRST, cards.DH_FIRST, cards.DH_FIRST]))
        # p1: [GZ], p2: [], p3: []; draw: [SPR_2, P, POW, FC]; discard: [DH]
        # d1: CF, d2: JF, d3: SPR, d4: SPR_2; draw: [P, POW, FC]; discard: [DH]
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.SPR_FIRST, cards.CF_FIRST, cards.CF_FIRST]))
        # Give GZ to player 1
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_GIVE_CARD, cards.GZ_FIRST))
        # p1: [SPR], p2: [GZ], p3: []; draw: [P, POW, FC]; discard: [DH, CF]
        # d1: P, d2: JF, d3: POW, d4: SPR_2; draw: [FC]; discard: [DH, CF]
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.SPR_2, cards.POW_FIRST, cards.POW_FIRST]))

        test_game.game_state = GoNutsGameState.GIVE_CARD

        # legal actions are position minus the SPR card we just got.
        # because giving away any SPR card is equivalent, we just remove from the selection the SPR with the lowest ID
        expected_legal_actions = np.zeros(translator.action_space_size())
        expected_legal_actions[TestHelpers.step_action(actions.ACTION_GIVE_CARD, cards.SPR_2)] = 1
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

    def test_legal_actions_include_only_last_picked_sprinkled_when_no_position_cards_in_give_card_state(self):

        test_game = GoNutsGame(3)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_for_sprinkled())
        test_game.start_game()

        # d1: CF, d2: DH, d3: SPR, d4: GZ; draw: [JF, MB, P, POW, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.SPR_FIRST, cards.DH_FIRST, cards.GZ_FIRST]))
        
        # legal actions are ONLY the SPR card
        expected_legal_actions = np.zeros(translator.action_space_size())
        expected_legal_actions[TestHelpers.step_action(actions.ACTION_GIVE_CARD, cards.SPR_FIRST)] = 1
        
        assert (translator.get_legal_actions(0) == expected_legal_actions).all()

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
        legal_actions = np.zeros(translator.action_space_size())
        for i in self.fixture_card_order()[:5]:
            legal_actions[i] = 1
        ret = np.append(ret, legal_actions)

        expected_observations = ret
        
        assert (translator.get_observations(0) == expected_observations).all()
        assert len(expected_observations) == 635

    def test_observations_correct_for_3_player_started_game(self):

        no_players = 3
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
        legal_actions = np.zeros(translator.action_space_size())
        for i in self.fixture_card_order()[:4]:
            legal_actions[i] = 1
        ret = np.append(ret, legal_actions)

        expected_observations = ret
        
        assert (translator.get_observations(0) == expected_observations).all()
        assert len(expected_observations) == 635

    def observation_comparer(self, actual, expected):

        assert (actual[obvs.OBVS_POSITIONS_START:obvs.OBVS_ALL_DISCARD_START] == expected[obvs.OBVS_POSITIONS_START:obvs.OBVS_ALL_DISCARD_START]).all()
        assert (actual[obvs.OBVS_ALL_DISCARD_START:obvs.OBVS_TOP_DISCARD_START] == expected[obvs.OBVS_ALL_DISCARD_START:obvs.OBVS_TOP_DISCARD_START]).all()
        assert (actual[obvs.OBVS_TOP_DISCARD_START:obvs.OBVS_SCORES_START] == expected[obvs.OBVS_TOP_DISCARD_START:obvs.OBVS_SCORES_START]).all()
        assert (actual[obvs.OBVS_SCORES_START:obvs.OBVS_LEGAL_ACTIONS_START] == expected[obvs.OBVS_SCORES_START:obvs.OBVS_LEGAL_ACTIONS_START]).all()
        assert (actual[obvs.OBVS_LEGAL_ACTIONS_START:] == expected[obvs.OBVS_LEGAL_ACTIONS_START:]).all()
        assert (actual == expected).all()

    def test_observations_correct_for_game_after_one_turn_from_player_index_zero_perspective(self):

        no_players = 4
        test_game = GoNutsGame(no_players)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game() # deals top 5
        # d1: 0 CF, d2: 3 DH, d3: 9 ECL, d4: 12 G, d5: 17 JF; draw: [23 MB, 25 P, 32 PWD, 66 FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST, cards.GZ_FIRST]))
        # p1: [0 CF, 23 MB], p2: [], p3: [], p4: [12 G]; draw: [25 P, 32 PWD, 66 FC]
        test_game.reset_turn()
        # d1: 25 P, d2: 32 PWD, d3: 9 ECL, d4: 66 FC, d5: 17 JF; draw: []; discard: [3 DH]

        # Positions (from player 0 perspective)
        no_cards = translator.total_possible_cards
        obs = np.array([])
        # p0
        p0position = np.zeros(no_cards)
        p0position[cards.CF_FIRST] = 1
        p0position[cards.MB_FIRST] = 1
        obs = np.append(obs, p0position)
        # p1
        obs = np.append(obs, np.zeros(no_cards))
        # p2
        obs = np.append(obs, np.zeros(no_cards))
        # p3
        p3position = np.zeros(no_cards)
        p3position[cards.GZ_FIRST] = 1
        obs = np.append(obs, p3position)
        # p4 (not in the game but included in the obvs space)
        obs = np.append(obs, np.zeros(no_cards))
        ret = obs

        # Discard
        discard = np.zeros(translator.total_possible_cards)
        discard[cards.DH_FIRST] = 1
        ret = np.append(ret, discard)
        ret = np.append(ret, discard)

        # Scores
        ret = np.append(ret, np.array([0, 0, 0, 2 / test_game.max_score, 0]))
        
        # Legal actions
        legal_actions = np.zeros(translator.action_space_size())
        legal_actions[cards.P_FIRST] = 1
        legal_actions[cards.POW_FIRST] = 1
        legal_actions[cards.ECL_FIRST] = 1
        legal_actions[cards.FC_FIRST] = 1
        legal_actions[cards.JF_FIRST] = 1
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
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST, cards.CF_FIRST]))
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
        discard[cards.DH_FIRST] = 1
        discard[cards.CF_FIRST] = 1
        ret = np.append(ret, discard)
        # Top
        discard = np.zeros(translator.total_possible_cards)
        discard[cards.DH_FIRST] = 1
        ret = np.append(ret, discard)

        # Scores
        ret = np.append(ret, np.array([0, 0, 0, 0, 0]))
        
        # Legal actions
        legal_actions = np.zeros(translator.action_space_size())
        legal_actions[cards.MB_FIRST] = 1
        legal_actions[cards.P_FIRST] = 1
        legal_actions[cards.ECL_FIRST] = 1
        legal_actions[cards.GZ_FIRST] = 1
        legal_actions[cards.JF_FIRST] = 1
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
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST, cards.GZ_FIRST]))
        # p1: [0 CF, 23 MB], p2: [], p3: [], p4: [12 G]; draw: [25 P, 32 PWD, 66 FC]
        test_game.reset_turn()
        # d1: 25 P, d2: 32 PWD, d3: 9 ECL, d4: 66 FC, d5: 17 JF; draw: []; discard: [3 DH]

        # Positions (from player 3 perspective)
        no_cards = translator.total_possible_cards
        obs = np.array([])
        # p3
        p3position = np.zeros(no_cards)
        p3position[cards.GZ_FIRST] = 1
        obs = np.append(obs, p3position)
        # p4 (not in the game but included in the obvs space)
        obs = np.append(obs, np.zeros(no_cards))
        # p0
        p0position = np.zeros(no_cards)
        p0position[cards.CF_FIRST] = 1
        p0position[cards.MB_FIRST] = 1
        obs = np.append(obs, p0position)
        # p1
        obs = np.append(obs, np.zeros(no_cards))
        # p2
        obs = np.append(obs, np.zeros(no_cards))
        
        ret = obs

        # Discard
        discard = np.zeros(translator.total_possible_cards)
        discard[cards.DH_FIRST] = 1
        ret = np.append(ret, discard)
        ret = np.append(ret, discard)

        # Scores
        ret = np.append(ret, np.array([2 / test_game.max_score, 0, 0, 0, 0]))
        
        # Legal actions
        legal_actions = np.zeros(translator.action_space_size())
        legal_actions[cards.P_FIRST] = 1
        legal_actions[cards.POW_FIRST] = 1
        legal_actions[cards.ECL_FIRST] = 1
        legal_actions[cards.FC_FIRST] = 1
        legal_actions[cards.JF_FIRST] = 1
        ret = np.append(ret, legal_actions)

        expected_observations = ret
        
        self.observation_comparer(translator.get_observations(3), expected_observations)

class TestHelpers:

    @staticmethod
    def step_action(action_type, action_id):
        
        return action_type + action_id

    @staticmethod
    def step_actions(action_type, action_ids):
        
        return [action_type + id for id in action_ids]

class TestGoNutsForDonutsGame:
    
    def fixture_card_filter(self):
        return [ cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.MB_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

    def fixture_card_order(self):
        return [ cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.MB_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

    def fixture_card_filter_with_red_velvet(self):
        return [ cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.RV_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

    def fixture_card_filter_with_double_chocolate(self):
        return [ cards.CF_FIRST, cards.DC_FIRST, cards.ECL_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.RV_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

    def fixture_card_filter_with_sprinkled(self):
        return [ cards.SPR_FIRST, cards.GZ_FIRST, cards.DH_FIRST, cards.GZ_2, cards.JF_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST, cards.JF_2 ]

    def fixture_card_filter_with_double_chocolate_and_no_chocolate_frosted(self):
        return [ cards.GZ_FIRST, cards.DC_FIRST, cards.ECL_FIRST, cards.GZ_2, cards.JF_FIRST, cards.RV_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.FC_FIRST ]

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

        assert test_game.donut_decks[0].card.id == cards.CF_FIRST
        assert test_game.donut_decks[0].card.symbol == 'CF'

        assert test_game.donut_decks[1].card.id == cards.DH_FIRST
        assert test_game.donut_decks[1].card.symbol == 'DH'

        assert test_game.donut_decks[2].card.id == cards.ECL_FIRST
        assert test_game.donut_decks[2].card.symbol == 'ECL'

        assert test_game.donut_decks[3].card.id == cards.GZ_FIRST
        assert test_game.donut_decks[3].card.symbol == 'GZ'

        assert test_game.donut_decks[4].card.id == cards.JF_FIRST
        assert test_game.donut_decks[4].card.symbol == 'JF'
    
    def test_uncontested_picks_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        test_game.pick_cards([cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST])

        assert test_game.players[0].position.cards[0].id == cards.CF_FIRST
        assert test_game.players[1].position.cards[0].id == cards.DH_FIRST
        assert test_game.players[2].position.cards[0].id == cards.ECL_FIRST
        assert test_game.players[3].position.cards[0].id == cards.GZ_FIRST

    def test_2nd_round_picks_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_order=self.fixture_card_order())
        test_game.start_game()

        test_game.pick_cards([cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST])
        test_game.reset_turn()

        test_game.pick_cards([cards.JF_FIRST, cards.MB_FIRST, cards.P_FIRST, cards.POW_FIRST])

        assert test_game.players[0].position.cards[1].id == cards.JF_FIRST
        assert test_game.players[1].position.cards[1].id == cards.MB_FIRST
        assert test_game.players[2].position.cards[1].id == cards.P_FIRST
        assert test_game.players[3].position.cards[1].id == cards.POW_FIRST

    def test_game_ends_when_not_enough_donuts_to_refill_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        # 5 out, 4 in deck
        test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST])
        test_game.reset_turn()
        # 2 removed, 5 out, 2 in deck
        test_game.pick_cards([cards.ECL_FIRST, cards.GZ_FIRST, cards.JF_FIRST, cards.JF_FIRST])
        test_game.reset_turn()
        # 3 removed, 5 out, -1 in deck

        assert test_game.is_game_over() == True

    def test_game_is_not_over_when_donuts_refilled_but_zero_draw_left(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        # 5 out, 4 in deck
        test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST])
        test_game.reset_turn()
        # 2 removed, 5 out, 2 in deck
        test_game.pick_cards([cards.ECL_FIRST, cards.JF_FIRST, cards.JF_FIRST, cards.JF_FIRST])
        test_game.reset_turn()
        # 2 removed, 5 out, 0 in deck

        assert test_game.is_game_over() == False

    def test_empty_decks_get_refilled(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        test_game.pick_cards([cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST])
        test_game.reset_turn()

        assert test_game.donut_decks[0].card.id == cards.MB_FIRST
        assert test_game.donut_decks[0].card.symbol == 'MB'

        assert test_game.donut_decks[1].card.id == cards.P_FIRST
        assert test_game.donut_decks[1].card.symbol == 'P'

        assert test_game.donut_decks[2].card.id == cards.POW_FIRST
        assert test_game.donut_decks[2].card.symbol == 'POW'

        assert test_game.donut_decks[3].card.id == cards.FC_FIRST
        assert test_game.donut_decks[3].card.symbol == 'FC'

    def test_contested_picks_dont_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.ECL_FIRST, cards.ECL_FIRST])

        assert len(test_game.players[0].position.cards) == 0
        assert len(test_game.players[1].position.cards) == 0
        assert len(test_game.players[2].position.cards) == 0
        assert len(test_game.players[3].position.cards) == 0

    def test_cards_picked_or_None_returned_by_pick_cards(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        cards_picked = test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.ECL_FIRST, cards.GZ_FIRST])

        assert cards_picked[0] == None
        assert cards_picked[1] == None
        assert cards_picked[2].id == cards.ECL_FIRST
        assert cards_picked[3].id == cards.GZ_FIRST

    def test_empty_decks_get_refilled_partial_picks(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()

        test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.ECL_FIRST, cards.GZ_FIRST])
        test_game.reset_turn()

        # Deck 1 - 8 - discarded and refilled
        assert test_game.donut_decks[0].card.id == cards.MB_FIRST
        assert test_game.donut_decks[0].card.symbol == 'MB'

        # Deck 2 - 7 - retained
        assert test_game.donut_decks[1].card.id == cards.DH_FIRST
        assert test_game.donut_decks[1].card.symbol == 'DH'

        # Deck 3 - 6 - taken and refilled
        assert test_game.donut_decks[2].card.id == cards.P_FIRST
        assert test_game.donut_decks[2].card.symbol == 'P'

        # Deck 4 - 5 - taken and refilled
        assert test_game.donut_decks[3].card.id == cards.POW_FIRST
        assert test_game.donut_decks[3].card.symbol == 'POW'

    def test_discard_pile_populated_with_contested_picks(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())

        test_game.start_game()

        test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST])
        test_game.reset_turn()

        assert len(test_game.discard.cards) == 2
        assert test_game.discard.cards[0].id == cards.CF_FIRST
        assert test_game.discard.cards[1].id == cards.DH_FIRST
        assert test_game.discard.peek_one().id == cards.DH_FIRST

    def test_card_action_chocolate_frosted_draws_one_from_deck(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game() # deals top 5

        test_game.card_action_chocolate_frosted(player_no=0)

        # first from top of deck is cards.MB_FIRST
        assert test_game.players[0].position.cards[0].id == cards.MB_FIRST
        # 3 cards left in deck
        assert test_game.deck.size() == 3

    def test_card_action_chocolate_frosted_draws_none_from_empty_deck(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game()
        # deck 9, deal 5, redeal 4, empty deck
        test_game.pick_cards([cards.CF_FIRST, cards.DH_FIRST, cards.ECL_FIRST, cards.GZ_FIRST])
        test_game.reset_turn()

        test_game.card_action_chocolate_frosted(0)

        # Just the pick, not the additional CF action
        assert test_game.players[0].position.size() == 1

    def test_card_action_eclair_draws_one_from_discard_pile(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game() # deals top 5
        test_game.pick_cards([cards.CF_FIRST, cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST]) # adds 8 then 7 to discard pile
        test_game.reset_turn()

        assert len(test_game.discard.cards) == 2

        test_game.card_action_eclair(0)

        # first from top of discard pile is card DH
        assert test_game.players[0].position.cards[0].id == cards.DH_FIRST
        assert len(test_game.discard.cards) == 1
        assert test_game.discard.cards[0].id == cards.CF_FIRST

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
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST, cards.GZ_FIRST]))
        # p1: [8 CF, 3 MB], p2: [], p3: [], p4: [5 G]; draw: [2 P, 1 PWD, 0 FC];
        test_game.reset_turn()
        # d1: 2 P, d2: 1 PWD, d3: 6 ECL, d4: 0 FC, d5: 4 JF; draw: []; discard: [7 DH]
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.JF_FIRST, cards.P_FIRST, cards.POW_FIRST, cards.ECL_FIRST]))
        # p1: [8 CF, 3 MB, 4 JF], p2: [2 P], p3: [1 PWD], p4: [5 G, 6 ECL, 7 DH]; draw: []; discard: []

        assert test_game.is_game_over() == True
        assert test_game.player_scores() == [ 0, 4, 3, 3 ]

        test_game.reset_turn()

    def test_players_advance_correctly_in_game_loop(self):

        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter())
        test_game.start_game() # deals top 5
        # d1: CF, d2: DH, d3: ECL, d4: G, d5: JF; draw: [MB, P, PWD, FC]; discard: []
        next_player_it0 = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.CF_FIRST))
        assert next_player_it0 == 1

        next_player_it1 = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.DH_FIRST))
        assert next_player_it1 == 2

        next_player_it2 = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.DH_FIRST))
        assert next_player_it2 == 3

        next_player_it3 = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.ECL_FIRST))
        assert next_player_it3 == 0

    def test_step_action_translate_for_discard(self):

        assert GoNutsGame.translate_step_action(GoNutsGameState.PICK_DISCARD, 3) == 3

    def test_step_action_translate_for_donut_pick(self):

        assert GoNutsGame.translate_step_action(GoNutsGameState.PICK_DONUT, 2) == 2

    def test_step_action_translate_for_pick_one_from_two(self):

        assert GoNutsGame.translate_step_action(GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS, 4) == 4

    def test_step_action_translate_for_give_card(self):

        assert GoNutsGame.translate_step_action(GoNutsGameState.GIVE_CARD, actions.ACTION_GIVE_CARD + 2) == 2

    def test_red_velvet_draws_correct_card_from_discard(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_with_red_velvet())
        test_game.start_game() # deals top 5
        # d1: CF, d2: DH, d3: ECL, d4: GZ, d5: JF; draw: [RV, P, PWD, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.DH_FIRST, cards.DH_FIRST, cards.CF_FIRST]))
        # p1: [], p2: [], p3: [], p4: []; draw: [RV, P, POW, FC]; discard: [CF, DH] (draw from back)
        # d1: RV, d2: P d3: 6 ECL, d4: GZ, d5: JF; draw: [POW, FC]; discard: [CF, DH] (draw from back)
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.P_FIRST))
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.RV_FIRST))
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.P_FIRST))
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.GZ_FIRST))
        # RV action requested for player 1
        assert next_player == 1

        # Pick CF as the choice from the discards
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DISCARD, cards.CF_FIRST))
        
        assert test_game.players[1].position.size() == 2
        assert test_game.players[1].position.cards[0].id == cards.RV_FIRST
        assert test_game.players[1].position.cards[1].id == cards.CF_FIRST

    def test_red_velvet_skips_actions_when_no_discard_cards(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_with_red_velvet())
        test_game.start_game() # deals top 5
        # d1: CF, d2: DH, d3: ECL, d4: GZ, d5: JF; draw: [RV, P, POW, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.JF_FIRST, cards.ECL_FIRST, cards.DH_FIRST, cards.GZ_FIRST]))
        # p1: [JF], p2: [ECL], p3: [DH], p4: [GZ]; draw: [RV, P, POW, FC]; discard: [] (draw from back)
        # d1: CF, d2: RV d3: 6 P, d4: POW, d5: FC; draw: []; discard: [] (draw from back)
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.P_FIRST))
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.RV_FIRST))
        test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.P_FIRST))
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.FC_FIRST))
        # No RV action because no cards in discard after end of turn 1
        assert next_player == 0

        assert test_game.players[1].position.size() == 2
        assert test_game.players[1].position.cards[0].id == cards.ECL_FIRST
        assert test_game.players[1].position.cards[1].id == cards.RV_FIRST

    def test_double_chocolate_draws_correct_card_from_deck(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_with_double_chocolate())
        test_game.start_game() # deals top 5
        # d1: CF, d2: DC, d3: ECL, d4: GZ, d5: JF; draw: [RV, P, PWD, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.DC_FIRST, cards.CF_FIRST]))
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.CF_FIRST))
        # After all donuts are picked, the game expects the DC player to choose from the top 2 deck cards
        assert next_player == 1
        assert test_game.game_state == GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS
        # deck cards: [RV, P]
        # choose P
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DECK, cards.P_FIRST))
        # Return to picking donuts
        assert next_player == 0
        assert test_game.game_state == GoNutsGameState.PICK_DONUT

        # Check player 1 has correct position
        assert test_game.players[1].position.size() == 2
        assert test_game.players[1].position.cards[0].id == cards.DC_FIRST
        assert test_game.players[1].position.cards[1].id == cards.P_FIRST

    def test_double_chocolate_draws_correct_card_from_deck_when_only_one_card_left(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_with_double_chocolate())
        test_game.start_game() # deals top 5
        # d1: CF, d2: DC, d3: ECL, d4: GZ, d5: JF; draw: [RV, P, PWD, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.CF_FIRST, cards.ECL_FIRST, cards.CF_FIRST, cards.GZ_FIRST]))
        # d1: RV, d2: DC, d3: P, d4: PWD, d5: JF; draw: [FC]; discard: [CF]
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.DC_FIRST, cards.JF_FIRST, cards.JF_FIRST]))
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.JF_FIRST))
        # After all donuts are picked, the game expects the DC player to choose from the top 2 deck cards (but there's only one left)
        assert next_player == 0
        assert test_game.game_state == GoNutsGameState.PICK_ONE_FROM_TWO_DECK_CARDS
        # deck cards: [FC]
        # choose P
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DECK, cards.FC_FIRST))
        # Return to picking donuts
        assert next_player == 0
        assert test_game.game_state == GoNutsGameState.PICK_DONUT

        # Check player with DC has correct position
        assert test_game.players[0].position.size() == 2
        assert test_game.players[0].position.cards[0].id == cards.DC_FIRST
        assert test_game.players[0].position.cards[1].id == cards.FC_FIRST

    def test_double_chocolate_draws_correct_card_from_deck_when_only_no_cards_left(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_with_double_chocolate_and_no_chocolate_frosted())
        test_game.start_game() # deals top 5
        # d1: GZ, d2: DC, d3: ECL, d4: GZ, d5: JF; draw: [RV, P, PWD, FC]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.GZ_FIRST, cards.ECL_FIRST, cards.GZ_2, cards.JF_FIRST]))
        # d1: RV, d2: DC, d3: P, d4: PWD, d5: FC; draw: []; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.DC_FIRST, cards.FC_FIRST, cards.FC_FIRST]))
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.FC_FIRST))
        # After all donuts are picked, the game expects the DC player to choose from the top 2 deck cards. But there are no cards to pick from, so the action is skipped
        # Return to picking donuts
        assert next_player == 0
        assert test_game.game_state == GoNutsGameState.PICK_DONUT

        # Check player with DC has correct position
        assert test_game.players[0].position.size() == 2
        assert test_game.players[0].position.cards[0].id == cards.GZ_FIRST
        assert test_game.players[0].position.cards[1].id == cards.DC_FIRST

    def test_sprinkled_gives_a_card_when_position_has_more_than_zero_cards(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_with_sprinkled())
        test_game.start_game() # deals top 5
        # d1: SPR, d2: GZ, d3: DH, d4: GZ_2, d5: JF; draw: [P, POW, FC, JF_2]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.GZ_FIRST, cards.DH_FIRST, cards.GZ_2, cards.JF_FIRST]))
        # p1: [GZ], p2: [DH], p3: [GZ], p4: [JF]; draw: [P, POW, FC, JF_2]; discard: []

        # d1: SPR, d2: P, d3: POW, d4: FC, d5: JF_2; draw: []; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.FC_FIRST, cards.SPR_FIRST, cards.FC_FIRST]))
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.FC_FIRST))
        # After all donuts are picked, move to SPR action
        assert next_player == 1
        assert test_game.game_state == GoNutsGameState.GIVE_CARD

        # choose GZ_FIRST, gives to player with lowest score, lowest id to break ties
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_GIVE_CARD, cards.DH_FIRST))
        # Return to picking donuts

        # Check player with SPR has correct position
        assert test_game.players[1].position.size() == 1
        assert test_game.players[1].position.cards[0].id == cards.SPR_FIRST

        # Check player receiving card has correct position (player 3, has lowest score with JF)
        assert test_game.players[3].position.size() == 2
        assert test_game.players[3].position.cards[0].id == cards.JF_FIRST
        assert test_game.players[3].position.cards[1].id == cards.DH_FIRST

    def test_sprinkled_gives_spr_card_when_position_has_zero_cards(self):
        test_game = GoNutsGame(4)

        test_game.setup_game(shuffle=False, deck_filter=self.fixture_card_filter_with_sprinkled())
        test_game.start_game() # deals top 5
        # d1: SPR, d2: GZ, d3: DH, d4: GZ_2, d5: JF; draw: [P, POW, FC, JF_2]; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.GZ_FIRST, cards.GZ_FIRST, cards.GZ_2, cards.JF_FIRST]))
        # p1: [], p2: [], p3: [GZ], p4: [JF]; draw: [P, POW, FC, JF_2]; discard: []
        # d1: SPR, d2: P, d3: DH, d4: POW, d5: FC; draw: []; discard: []
        test_game.execute_game_loop_with_actions(TestHelpers.step_actions(actions.ACTION_DONUT, [cards.FC_FIRST, cards.SPR_FIRST, cards.FC_FIRST]))
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_DONUT, cards.FC_FIRST))
        # After all donuts are picked, move to SPR action
        assert next_player == 1
        assert test_game.game_state == GoNutsGameState.GIVE_CARD

        # must choose SPR_FIRST, gives to player with lowest score, lowest id to break ties
        next_player = test_game.execute_game_loop(TestHelpers.step_action(actions.ACTION_GIVE_CARD, cards.SPR_FIRST))

        # Check player using SPR has correct position
        assert test_game.players[1].position.size() == 0

        # Check player receiving card has correct position (player 0, has lowest score, lowest ID to break ties)
        assert test_game.players[0].position.size() == 1
        assert test_game.players[0].position.cards[0].id == cards.SPR_FIRST