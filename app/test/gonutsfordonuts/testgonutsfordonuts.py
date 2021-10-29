from gonutsfordonuts.envs.gonutsfordonuts import GoNutsGame
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutDeckPosition, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered

class TestGoNutsForDonuts:
    
    def fixture_contents(self):
        contents = [
          {'card': ChocolateFrosted, 'info': {}, 'count': 1}  #0 
           ,  {'card': DonutHoles, 'info': {}, 'count':  1} #1 
        ,  {'card': Eclair, 'info': {}, 'count':  1}  #2   
          ,  {'card': Glazed, 'info': {}, 'count':  1} #3  
           ,  {'card': JellyFilled, 'info': {}, 'count':  1} #4 
           ,  {'card': MapleBar,  'info': {}, 'count':  1} #5 
           ,  {'card': Plain, 'info': {}, 'count':  1} #6 
          ,  {'card': Powdered, 'info': {}, 'count':  1}  #7 
          ,  {'card': FrenchCruller, 'info': {}, 'count': 1}  #8 (last due to variability) 
        ]
        
        contents.reverse() # (deck is a stack, so reverse so card_id = 1 is on top)
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

        
    