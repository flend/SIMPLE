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
       
class Deck():
    def __init__(self, contents):
        self.contents = contents
        self.create()
    
    def shuffle(self):
        random.shuffle(self.cards)

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

    def create(self):
        self.cards = []

        card_id = 0
        for order, x in enumerate(self.contents):
            x['info']['order'] = order
            for i in range(x['count']):
                x['info']['id'] = card_id
                self.add([x['card'](**x['info'])])
                card_id += 1
                                
    def size(self):
        return len(self.cards)

class DonutDeckPosition():
    def __init__(self, card):
        self.card = card
        self.taken = False
        self.to_discard = False

    def get_card_type(self):
        return self.card.type

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
