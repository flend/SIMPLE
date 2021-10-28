from gonutsfordonuts.envs.gonutsfordonuts import GoNutsGame
from gonutsfordonuts.envs.classes import ChocolateFrosted, DonutDeckPosition, DonutHoles, Eclair, FrenchCruller, Glazed, JellyFilled, MapleBar, Plain, Powdered

class TestGoNutsForDonuts:
    
    def fixture_contents(self):
        return [
          {'card': ChocolateFrosted, 'info': {}, 'count': 1}  #0 
           ,  {'card': DonutHoles, 'info': {}, 'count':  1} #1 
        ,  {'card': Eclair, 'info': {}, 'count':  1}  #2   
          ,  {'card': Glazed, 'info': {}, 'count':  1} #3  
           ,  {'card': JellyFilled, 'info': {}, 'count':  1} #4 
           ,  {'card': MapleBar,  'info': {}, 'count':  1} #5 
           ,  {'card': Plain, 'info': {}, 'count':  1} #6 
          ,  {'card': Powdered, 'info': {}, 'count':  1}  #7 
          ,  {'card': FrenchCruller, 'info': {}, 'count': 1}  #8 (last due to variability) 
        ].reverse() # (deck is a stack, so reverse so card_id = 1 is on top)

    def test_standard_deck_contents_size(self):

        test_game = GoNutsGame(2)
        test_game.setup_game()
        assert test_game.deck.size() == 37

    def test_override_deck_contents_size(self):

        test_game = GoNutsGame(2)
        test_game.setup_game(self.fixture_contents())
        assert test_game.deck.size() == 8

    def test_decks_setup_for_first_turn(self):

        test_game = GoNutsGame(4)
        test_game.setup_game(self.fixture_contents(), False)

        test_game.reset_turn()

        assert test_game.donut_decks[0].card.id == 1
        assert test_game.donut_decks[1].card.id == 2
        assert test_game.donut_decks[2].card.id == 3
        assert test_game.donut_decks[3].card.id == 4
        
    