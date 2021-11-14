import numpy as np

from gonutsfordonuts.envs.gonutsfordonuts import GoNutsGame, GoNutsScorer, GoNutsGameGymTranslator
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered
from gonutsfordonuts.envs.classes import Position

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

class TestGoNutsForDonutsGymTranslator:

    def fixture_contents(self):
        contents = [
          {'card': ChocolateFrosted, 'info': {}, 'count': 1}  #0 / card.id = 8
           ,  {'card': DonutHoles, 'info': {}, 'count':  1} #1 / card.id = 7
        ,  {'card': Eclair, 'info': {}, 'count':  1}  #2 / card.id = 6
          ,  {'card': Glazed, 'info': {}, 'count':  1} #3 / card.id = 5
           ,  {'card': JellyFilled, 'info': {}, 'count':  1} #4 / card.id = 4
           ,  {'card': MapleBar,  'info': {}, 'count':  1} #5 / card.id = 3
           ,  {'card': Plain, 'info': {}, 'count':  1} #6 / card.id = 2
          ,  {'card': Powdered, 'info': {}, 'count':  1}  #7 / card.id = 1
          ,  {'card': FrenchCruller, 'info': {}, 'count': 1}  #8 / card.id = 0 (last due to variability) 
        ]
        
        contents.reverse() # (deck is a stack, so reverse so card #0 is on top - note it gets card.id = 8)
        return contents

    def test_legal_actions_have_only_donut_deck_picks(self):

        test_game = GoNutsGame(4)
        translator = GoNutsGameGymTranslator(test_game)

        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.start_game()

        # 5 dealt cards in donut array
        expected_legal_actions = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        assert (translator.get_legal_actions() == expected_legal_actions).all()

class TestGoNutsForDonuts:
    
    def fixture_contents(self):
        contents = [
          {'card': ChocolateFrosted, 'info': {}, 'count': 1}  #0 / card.id = 8
           ,  {'card': DonutHoles, 'info': {}, 'count':  1} #1 / card.id = 7
        ,  {'card': Eclair, 'info': {}, 'count':  1}  #2 / card.id = 6
          ,  {'card': Glazed, 'info': {}, 'count':  1} #3 / card.id = 5
           ,  {'card': JellyFilled, 'info': {}, 'count':  1} #4 / card.id = 4
           ,  {'card': MapleBar,  'info': {}, 'count':  1} #5 / card.id = 3
           ,  {'card': Plain, 'info': {}, 'count':  1} #6 / card.id = 2
          ,  {'card': Powdered, 'info': {}, 'count':  1}  #7 / card.id = 1
          ,  {'card': FrenchCruller, 'info': {}, 'count': 1}  #8 / card.id = 0 (last due to variability) 
        ]
        
        contents.reverse() # (deck is a stack, so reverse so card #0 is on top - note it gets card.id = 8)
        return contents

    def test_standard_deck_contents_size(self):

        test_game = GoNutsGame(2)
        test_game.setup_game()
        assert test_game.deck.size() == 37

    def test_override_deck_contents_size(self):

        test_game = GoNutsGame(2)
        test_game.setup_game(deck_contents=self.fixture_contents())
        assert test_game.deck.size() == 9

    def test_decks_setup_for_first_turn(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        assert test_game.donut_decks[0].card.id == 8
        assert test_game.donut_decks[0].card.symbol == 'CF'

        assert test_game.donut_decks[1].card.id == 7
        assert test_game.donut_decks[1].card.symbol == 'DH'

        assert test_game.donut_decks[2].card.id == 6
        assert test_game.donut_decks[2].card.symbol == 'ECL'

        assert test_game.donut_decks[3].card.id == 5
        assert test_game.donut_decks[3].card.symbol == 'GZ'

        assert test_game.donut_decks[4].card.id == 4
        assert test_game.donut_decks[4].card.symbol == 'JF'
    
    def test_uncontested_picks_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        test_game.pick_cards([8, 7, 6, 5])

        assert test_game.players[0].position.cards[0].id == 8
        assert test_game.players[1].position.cards[0].id == 7
        assert test_game.players[2].position.cards[0].id == 6
        assert test_game.players[3].position.cards[0].id == 5

    def test_2nd_round_picks_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        test_game.pick_cards([8, 7, 6, 5])
        test_game.reset_turn()

        test_game.pick_cards([4, 3, 2, 1])

        assert test_game.players[0].position.cards[1].id == 4
        assert test_game.players[1].position.cards[1].id == 3
        assert test_game.players[2].position.cards[1].id == 2
        assert test_game.players[3].position.cards[1].id == 1

    def test_game_ends_when_not_enough_donuts_to_refill_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        # 5 out, 4 in deck
        test_game.pick_cards([8, 8, 7, 7])
        test_game.reset_turn()
        # 2 removed, 5 out, 2 in deck
        test_game.pick_cards([6, 5, 4, 4])
        test_game.reset_turn()
        # 3 removed, 5 out, -1 in deck

        assert test_game.is_game_over() == True

    def test_game_is_not_over_when_donuts_refilled_but_zero_draw_left(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        # 5 out, 4 in deck
        test_game.pick_cards([8, 8, 7, 7])
        test_game.reset_turn()
        # 2 removed, 5 out, 2 in deck
        test_game.pick_cards([6, 4, 4, 4])
        test_game.reset_turn()
        # 2 removed, 5 out, 0 in deck

        assert test_game.is_game_over() == False

    def test_empty_decks_get_refilled(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        test_game.pick_cards([8, 7, 6, 5])
        test_game.reset_turn()

        assert test_game.donut_decks[0].card.id == 3
        assert test_game.donut_decks[0].card.symbol == 'MB'

        assert test_game.donut_decks[1].card.id == 2
        assert test_game.donut_decks[1].card.symbol == 'P'

        assert test_game.donut_decks[2].card.id == 1
        assert test_game.donut_decks[2].card.symbol == 'POW'

        assert test_game.donut_decks[3].card.id == 0
        assert test_game.donut_decks[3].card.symbol == 'FC'

    def test_contested_picks_dont_go_to_positions(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        test_game.pick_cards([8, 8, 6, 6])

        assert len(test_game.players[0].position.cards) == 0
        assert len(test_game.players[1].position.cards) == 0
        assert len(test_game.players[2].position.cards) == 0
        assert len(test_game.players[3].position.cards) == 0

    def test_cards_picked_or_None_returned_by_pick_cards(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        cards_picked = test_game.pick_cards([8, 8, 6, 5])

        assert cards_picked[0] == None
        assert cards_picked[1] == None
        assert cards_picked[2].id == 6
        assert cards_picked[3].id == 5

    def test_empty_decks_get_refilled_partial_picks(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()

        test_game.pick_cards([8, 8, 6, 5])
        test_game.reset_turn()

        # Deck 1 - 8 - discarded and refilled
        assert test_game.donut_decks[0].card.id == 3
        assert test_game.donut_decks[0].card.symbol == 'MB'

        # Deck 2 - 7 - retained
        assert test_game.donut_decks[1].card.id == 7
        assert test_game.donut_decks[1].card.symbol == 'DH'

        # Deck 3 - 6 - taken and refilled
        assert test_game.donut_decks[2].card.id == 2
        assert test_game.donut_decks[2].card.symbol == 'P'

        # Deck 4 - 5 - taken and refilled
        assert test_game.donut_decks[3].card.id == 1
        assert test_game.donut_decks[3].card.symbol == 'POW'

    def test_discard_pile_populated_with_contested_picks(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)

        test_game.reset_turn()

        test_game.pick_cards([8, 8, 7, 7])
        test_game.reset_turn()

        assert len(test_game.discard.cards) == 2
        assert test_game.discard.cards[0].id == 8
        assert test_game.discard.cards[1].id == 7

    def test_card_action_chocolate_frosted_draws_one_from_deck(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn() # deals top 5

        test_game.card_action_chocolate_frosted(0)

        # first from top of deck is card id 3
        assert test_game.players[0].position.cards[0].id == 3
        # 3 cards left in deck
        assert test_game.deck.size() == 3

    def test_card_action_chocolate_frosted_draws_none_from_empty_deck(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn()
        # deck 9, deal 5, redeal 4, empty deck
        test_game.pick_cards([8, 7, 6, 5])
        test_game.reset_turn()

        test_game.card_action_chocolate_frosted(0)

        # Just the pick, not the additional CF action
        assert test_game.players[0].position.size() == 1

    def test_card_action_eclair_draws_one_from_discard_pile(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn() # deals top 5
        test_game.pick_cards([8, 8, 7, 7]) # adds 8 then 7 to discard pile
        test_game.reset_turn()

        assert len(test_game.discard.cards) == 2

        test_game.card_action_eclair(0)

        # first from top of discard pile is card id 7
        assert test_game.players[0].position.cards[0].id == 7
        assert len(test_game.discard.cards) == 1
        assert test_game.discard.cards[0].id == 8

    def test_card_action_eclair_draws_none_from_empty_discard_pile(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.reset_turn() # deals top 5

        test_game.card_action_eclair(0)

        assert test_game.players[0].position.size() == 0

    def test_full_game_scoring(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(deck_contents=self.fixture_contents(), shuffle=False)
        test_game.start_game() # deals top 5
        # d1: 8 CF, d2: 7 DH, d3: 6 ECL, d4: 5 G, d5: 4 JF; draw: [3 MB, 2 P, 1 PWD, 0 FC]; discard: []
        test_game.do_player_actions([8, 7, 7, 5])
        # p1: [8 CF, 3 MB], p2: [], p3: [], p4: [5 G]; draw: [2 P, 1 PWD, 0 FC];
        test_game.reset_turn()
        # d1: 2 P, d2: 1 PWD, d3: 6 ECL, d4: 0 FC, d5: 4 JF; draw: []; discard: [7 DH]
        test_game.do_player_actions([4, 2, 1, 6])
        # p1: [8 CF, 3 MB, 4 JF], p2: [2 P], p3: [1 PWD], p4: [5 G, 6 ECL, 7 DH]; draw: []; discard: []
        
        assert test_game.is_game_over() == True
        assert test_game.player_scores() == [ 0, 4, 3, 3 ]

        test_game.reset_turn()
    



        
    