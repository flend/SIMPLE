import random
from stable_baselines import logger
import gonutsfordonuts.envs.cards as cards

class Player():
    def __init__(self, id):
        self.id = id
        self.score = 0
        self.hand = Hand()
        self.position = Position()

class Card():
    def __init__(self, id, type, name):
        self.id = id
        self.type = type
        self.name = name

    def __eq__(self, __o: object) -> bool:
        if (isinstance(__o, Card)):
            return self.id == __o.id and self.type == __o.type and self.name == __o.name
        return False
        
class ChocolateFrosted(Card):
    def __init__(self, id):
        super(ChocolateFrosted, self).__init__(id, cards.TYPE_CF, 'chocolate_frosted')
        self.colour = 'green'
        self.symbol = 'CF'

class DonutHoles(Card):
    def __init__(self, id):
        super(DonutHoles, self).__init__(id, cards.TYPE_DH, 'donut_holes')
        self.colour = 'green'
        self.symbol = 'DH'
        
class Eclair(Card):
    def __init__(self, id):
        super(Eclair, self).__init__(id, cards.TYPE_ECL, 'eclair')
        self.colour = 'green'
        self.symbol = 'ECL'

class FrenchCruller(Card):
    def __init__(self, id):
        super(FrenchCruller, self).__init__(id, cards.TYPE_FC, 'french_cruller')
        self.colour = 'green'
        self.symbol = 'FC'

class Glazed(Card):
    def __init__(self, id):
        super(Glazed, self).__init__(id, cards.TYPE_GZ, 'glazed')
        self.colour = 'green'
        self.symbol = 'GZ'

class JellyFilled(Card):
    def __init__(self, id):
        super(JellyFilled, self).__init__(id, cards.TYPE_JF, 'jelly_filled')
        self.colour = 'green'
        self.symbol = 'JF'

class MapleBar(Card):
    def __init__(self, id):
        super(MapleBar, self).__init__(id, cards.TYPE_MB, 'maple_bar')
        self.colour = 'green'
        self.symbol = 'MB'

class Plain(Card):
    def __init__(self, id):
        super(Plain, self).__init__(id, cards.TYPE_P, 'plain')
        self.colour = 'green'
        self.symbol = 'P'

class Powdered(Card):
    def __init__(self, id):
        super(Powdered, self).__init__(id, cards.TYPE_POW, 'powdered')
        self.colour = 'green'
        self.symbol = 'POW'

class BostonCream(Card):
    def __init__(self, id):
        super(BostonCream, self).__init__(id, cards.TYPE_BC, 'boston_cream')
        self.colour = 'pink'
        self.symbol = 'BC'

class DoubleChocolate(Card):
    def __init__(self, id):
        super(DoubleChocolate, self).__init__(id, cards.TYPE_DC, 'double_chocolate')
        self.colour = 'pink'
        self.symbol = 'DC'

class RedVelvet(Card):
    def __init__(self, id):
        super(RedVelvet, self).__init__(id, cards.TYPE_RV, 'red_velvet')
        self.colour = 'pink'
        self.symbol = 'RV'

class Sprinkled(Card):
    def __init__(self, id):
        super(Sprinkled, self).__init__(id, cards.TYPE_SPR, 'sprinkled')
        self.colour = 'pink'
        self.symbol = 'SPR'

class BearClaw(Card):
    def __init__(self, id):
        super(BearClaw, self).__init__(id, cards.TYPE_BC, 'bear_claw')
        self.colour = 'purple'
        self.symbol = 'BEAR'

class CinnamonTwist(Card):
    def __init__(self, id):
        super(CinnamonTwist, self).__init__(id, cards.TYPE_CT, 'cinnamon_twist')
        self.colour = 'purple'
        self.symbol = 'CT'

class Coffee(Card):
    def __init__(self, id):
        super(Coffee, self).__init__(id, cards.TYPE_CF, 'coffee')
        self.colour = 'purple'
        self.symbol = 'CFF'

class DayOldDonuts(Card):
    def __init__(self, id):
        super(DayOldDonuts, self).__init__(id, cards.TYPE_DOD, 'day_old_donuts')
        self.colour = 'purple'
        self.symbol = 'DOD'

class Milk(Card):
    def __init__(self, id):
        super(Milk, self).__init__(id, cards.TYPE_MILK, 'milk')
        self.colour = 'purple'
        self.symbol = 'MLK'

class OldFashioned(Card):
    def __init__(self, id):
        super(OldFashioned, self).__init__(id, cards.TYPE_OLD, 'old_fashioned')
        self.colour = 'purple'
        self.symbol = 'OLD'

class MapleFrosted(Card):
    def __init__(self, id):
        super(MapleFrosted, self).__init__(id, cards.TYPE_MF, 'maple_frosted')
        self.colour = 'blue'
        self.symbol = 'MF'

class MuchoMatcha(Card):
    def __init__(self, id):
        super(MuchoMatcha, self).__init__(id, cards.TYPE_MM, 'mucho_matcha')
        self.colour = 'blue'
        self.symbol = 'MM'

class RaspberryFrosted(Card):
    def __init__(self, id):
        super(RaspberryFrosted, self).__init__(id, cards.TYPE_RF, 'raspberry_frosted')
        self.colour = 'blue'
        self.symbol = 'RF'

class StrawberryGlazed(Card):
    def __init__(self, id):
        super(StrawberryGlazed, self).__init__(id, cards.TYPE_SG, 'strawberry_glazed')
        self.colour = 'blue'
        self.symbol = 'SG'
       
class Deck():
    def __init__(self, players, contents):
        # The cards to use for this case
        self.contents = contents
        # The full set of standard cards, used to define the spaces
        self.cards = [] # Stack to pull cards from
        self.base_deck = [] # Snapshot of the full standard deck
        self.create()
    
    def shuffle(self):
        random.shuffle(self.cards)

    def reorder(self, new_order):
        """Reorders the deck with the order specified, any non specified cards follow in current ordering
           Order specified to-be is draw order
        """

        new_card_order = []

        # Put the requested cards first
        for i in new_order:
            new_card_order.append(self.cards[i])

        # Add any unrequested cards in their conventional order afterwards
        for card in self.cards:
            if not card.id in new_order:
                new_card_order.append(card)

        # Reverse the deck so that the chosen cards are drawn first
        new_card_order.reverse()

        self.cards = new_card_order

    def filter(self, filter):
        """Filters the deck to keep only the card ids in filter, in the order specified.
           Order specified to-be is draw order
        """
        new_deck = []
        for i in filter:
            new_deck.append(self.cards[i])

        # Reverse the deck so that the chosen cards are drawn first
        new_deck.reverse()

        self.cards = new_deck

    def draw(self, n):
        drawn = []
        for x in range(n):
            drawn.append(self.cards.pop())
        return drawn
    
    def draw_one(self):
        return self.cards.pop()

    def peek_one(self):
        if not len(self.cards):
            return None
        return self.cards[-1]

    def peek_in_nth_position(self, n):
        if len(self.cards) < n:
            return None
        return self.cards[-n]
    
    def remove_one(self, card):
        self.cards.remove(card)

    def add(self, cards):
        for card in cards:
            self.cards.append(card)

    def add_to_base_deck(self, cards):
        for card in cards:
            self.base_deck.append(card)

    def create(self):

        # card.id uniquely identifies the card in the deck
        # this occurs before shuffling, so a particular card id is always of a particular type
        # (if the deck construction does not change)
        # card.order is a numeric identifier of card type but this is unnecessary
        # since cards are objects that know what type they are anyway
        # TODO: remove order unless there's a good reason for it
        
        card_id = 0
        for order, x in enumerate(self.contents):
            for i in range(x['count']):
                x['info']['id'] = card_id
                card = [x['card'](**x['info'])]
                self.add(card)
                self.add_to_base_deck(card)
                card_id += 1
                                
    def size(self):
        return len(self.cards)

class DonutDeckPosition():
    def __init__(self, card):
        self.card = card
        self.taken = False
        self.to_discard = False

    def get_card(self):
        return self.card

    def set_taken(self):
        self.taken = True
    
    def set_to_discard(self):
        self.to_discard = True

class Hand():
    def __init__(self):
        self.cards = []  
    
    def add(self, cards):
        for card in cards:
            self.cards.append(card)
    
    def size(self):
        return len(self.cards)
    
    def pick(self, name):
        for i, c in enumerate(self.cards):
            if c.name == name:
                self.cards.pop(i)
                return c
        
                
class Discard():
    def __init__(self):
        self.cards = []  
    
    def add(self, cards):
        for card in cards:
            self.cards.append(card)
    
    def draw(self, n):
        drawn = []
        for x in range(n):
            drawn.append(self.cards.pop())
        return drawn

    def draw_one(self):
        return self.cards.pop()

    def remove_one(self, card):
        self.cards.remove(card)

    def peek_one(self):
        if not len(self.cards):
            return None
        return self.cards[-1]
    
    def size(self):
        return len(self.cards)
    
class Position():
    def __init__(self):
        self.cards = []  
    
    def add(self, cards):
        for card in cards:
            self.cards.append(card)

    def add_one(self, card):
        self.cards.append(card)
    
    def size(self):
        return len(self.cards)

    def pick(self, name):
        for i, c in enumerate(self.cards):
            if c.name == name:
                self.cards.pop(i)
                return c

    def remove_one(self, card):
        self.cards.remove(card)

    def contains_id(self, card_id):
        return card_id in [c.id for c in self.cards]
