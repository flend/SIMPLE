import random

class Player():
    def __init__(self, id):
        self.id = id
        self.score = 0
        self.hand = Hand()
        self.position = Position()

class Card():
    def __init__(self, id, order, name):
        self.id = id
        self.order = order
        self.name = name
        
class ChocolateFrosted(Card):
    def __init__(self, id, order):
        super(ChocolateFrosted, self).__init__(id, order, 'chocolate_frosted')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'CF'

class DonutHoles(Card):
    def __init__(self, id, order):
        super(DonutHoles, self).__init__(id, order, 'donut_holes')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'DH'
        
class Eclair(Card):
    def __init__(self, id, order):
        super(Eclair, self).__init__(id, order, 'eclair')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'ECL'

class FrenchCruller(Card):
    def __init__(self, id, order):
        super(FrenchCruller, self).__init__(id, order, 'french_cruller')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'FC'

class Glazed(Card):
    def __init__(self, id, order):
        super(Glazed, self).__init__(id, order, 'glazed')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'GZ'

class JellyFilled(Card):
    def __init__(self, id, order):
        super(JellyFilled, self).__init__(id, order, 'jelly_filled')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'JF'

class MapleBar(Card):
    def __init__(self, id, order):
        super(MapleBar, self).__init__(id, order, 'maple_bar')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'MB'

class Plain(Card):
    def __init__(self, id, order):
        super(Plain, self).__init__(id, order, 'plain')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'P'

class Powdered(Card):
    def __init__(self, id, order):
        super(Powdered, self).__init__(id, order, 'powdered')
        self.colour = 'green'
        self.type = self.name
        self.symbol = 'POW'

class BostonCream(Card):
    def __init__(self, id, order):
        super(BostonCream, self).__init__(id, order, 'boston_cream')
        self.colour = 'pink'
        self.type = self.name
        self.symbol = 'BC'

class DoubleChocolate(Card):
    def __init__(self, id, order):
        super(DoubleChocolate, self).__init__(id, order, 'double_chocolate')
        self.colour = 'pink'
        self.type = self.name
        self.symbol = 'DC'

class RedVelvet(Card):
    def __init__(self, id, order):
        super(RedVelvet, self).__init__(id, order, 'red_velvet')
        self.colour = 'pink'
        self.type = self.name
        self.symbol = 'RV'

class Sprinkled(Card):
    def __init__(self, id, order):
        super(Sprinkled, self).__init__(id, order, 'sprinkled')
        self.colour = 'pink'
        self.type = self.name
        self.symbol = 'SPR'

class BearClaw(Card):
    def __init__(self, id, order):
        super(BearClaw, self).__init__(id, order, 'bear_claw')
        self.colour = 'purple'
        self.type = self.name
        self.symbol = 'BEAR'

class CinnamonTwist(Card):
    def __init__(self, id, order):
        super(CinnamonTwist, self).__init__(id, order, 'cinnamon_twist')
        self.colour = 'purple'
        self.type = self.name
        self.symbol = 'CT'

class Coffee(Card):
    def __init__(self, id, order):
        super(Coffee, self).__init__(id, order, 'coffee')
        self.colour = 'purple'
        self.type = self.name
        self.symbol = 'CFF'

class DayOldDonuts(Card):
    def __init__(self, id, order):
        super(DayOldDonuts, self).__init__(id, order, 'day_old_donuts')
        self.colour = 'purple'
        self.type = self.name
        self.symbol = 'DOD'

class Milk(Card):
    def __init__(self, id, order):
        super(Milk, self).__init__(id, order, 'milk')
        self.colour = 'purple'
        self.type = self.name
        self.symbol = 'MLK'

class OldFashioned(Card):
    def __init__(self, id, order):
        super(OldFashioned, self).__init__(id, order, 'old_fashioned')
        self.colour = 'purple'
        self.type = self.name
        self.symbol = 'OLD'

class MapleFrosted(Card):
    def __init__(self, id, order):
        super(MapleFrosted, self).__init__(id, order, 'maple_frosted')
        self.colour = 'blue'
        self.type = self.name
        self.symbol = 'MF'

class MuchoMatcha(Card):
    def __init__(self, id, order):
        super(MuchoMatcha, self).__init__(id, order, 'mucho_matcha')
        self.colour = 'blue'
        self.type = self.name
        self.symbol = 'MM'

class RaspberryFrosted(Card):
    def __init__(self, id, order):
        super(RaspberryFrosted, self).__init__(id, order, 'raspberry_frosted')
        self.colour = 'blue'
        self.type = self.name
        self.symbol = 'RF'

class StrawberryGlazed(Card):
    def __init__(self, id, order):
        super(StrawberryGlazed, self).__init__(id, order, 'strawberry_glazed')
        self.colour = 'blue'
        self.type = self.name
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

    def draw(self, n):
        drawn = []
        for x in range(n):
            drawn.append(self.cards.pop())
        return drawn
    
    def draw_one(self):
        return self.cards.pop()
    
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
            x['info']['order'] = order
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
