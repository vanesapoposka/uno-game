import random
COLORS = ["R","G","B","Y"]
def build_deck():
    deck = []
    for color in COLORS:
        deck.append(("N", color, "0"))
        for val in range(1, 10):
            deck.append(("N", color, str(val)))
            deck.append(("N", color, str(val)))
        for val in ["Skip", "Reverse", "Draw2"]:
            deck.append(("A", color, val))
            deck.append(("A", color, val))

    for _ in range(4):
        deck.append(("W", None, "Wild"))
        deck.append(("W", None, "Wild4"))

    random.shuffle(deck)
    return deck

if __name__ == "__main__":
    human = [('W', None, 'Wild'), ('W', None, 'Wild4'), ('A', 'Y', 'Skip'), ('A', 'G', 'Reverse'), ('A', 'B', 'Draw2'),
             ('A', 'R', 'Draw2'), ('A', 'Y', 'Skip')]
    ai = [('W', None, 'Wild'), ('W', None, 'Wild4'), ('A', 'R', 'Reverse'), ('A', 'G', 'Skip'), ('A', 'B', 'Draw2'),
          ('A', 'Y', 'Draw2'), ('A', 'R', 'Skip')]
    top_card = [('N', 'G', '5')]
    deck = build_deck()

    cards = human + ai + top_card
    for card in cards:
        if card in deck:
            idx = deck.index(card)
            deck.pop(idx)

    print(deck)
