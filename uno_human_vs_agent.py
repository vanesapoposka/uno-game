import pygame
import random
import sys
from test_cases import get_test_case
import time
import math
from collections import defaultdict

SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = 60
CARD_WIDTH, CARD_HEIGHT = 90, 130
CARD_SPACING = 18
HAND_Y_HUMAN = SCREEN_HEIGHT - CARD_HEIGHT - 30
HAND_Y_AI = 30
CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

AI_THINK_DELAY_MS = 350
AI_ALGORITHM = "minimax"
SEARCH_DEPTH = 4

COLORS = ["R", "G", "B", "Y"]

MAX_HAND_WIDTH = SCREEN_WIDTH - 100  # max width for hand display
SCROLL_SPEED = 30  # pixels to scroll per key press
MOUSE_SCROLL_SENSITIVITY = 1.5  # mouse wheel scroll sensitivity

# to build the deck of cards used for the game with random shuffle
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

# to check if the player's card is playable
def is_playable(curr_card, top_card, wild_card_color=None):
    if top_card is None:
        return True

    curr_type, curr_color, curr_value = curr_card
    _, top_color, top_value = top_card

    if curr_type == "W":
        return True

    if wild_card_color is not None:
        return curr_color == wild_card_color

    return curr_value == top_value or curr_color == top_color

# class to initiate the game's state
class Game:
    # constructor function
    def __init__(
            self,
            draw_deck=None,
            played_deck=None,
            players_decks=None,
            current_player=0,
            wild_card_color=None,
            winner=None,
            possible_opponent_cards=None,
    ):
        self.draw_deck = draw_deck if draw_deck is not None else []
        self.played_deck = played_deck if played_deck is not None else []
        self.players_decks = players_decks if players_decks is not None else [[], []]
        self.current_player = current_player
        self.wild_card_color = wild_card_color
        self.winner = winner
        self.possible_opponent_cards = possible_opponent_cards if possible_opponent_cards is not None else []

    # copy function
    def copy(self):
        return Game(
            draw_deck=self.draw_deck[:],
            played_deck=self.played_deck[:],
            players_decks=[deck[:] for deck in self.players_decks],
            current_player=self.current_player,
            wild_card_color=self.wild_card_color,
            winner=self.winner,
            possible_opponent_cards=self.possible_opponent_cards[:]
        )

    # to see who's the next player
    def next_player(self, player=None):
        if player is None:
            player = self.current_player
        return 1 - player

    # to draw a card from the draw deck
    def draw_card(self, player):
        if not self.draw_deck:
            if len(self.played_deck) > 1:
                top = self.played_deck.pop()
                cards = self.played_deck[:]
                random.shuffle(cards)
                self.draw_deck = cards
                self.played_deck = [top]
        if self.draw_deck:
            self.players_decks[player].append(self.draw_deck.pop())

    # to see the top card from played deck
    def top_card(self):
        return self.played_deck[-1] if self.played_deck else None

    # to return the card color that is most common among the player's cards
    def colored_cards(self):
        arr_colors = [0] * 4
        for card in self.players_decks[self.current_player]:
            t, c, v = card
            if c is not None:
                if c == "R": arr_colors[0] += 1
                if c == "G": arr_colors[1] += 1
                if c == "B": arr_colors[2] += 1
                if c == "Y": arr_colors[3] += 1

        if arr_colors[0] == arr_colors[1] == arr_colors[2] == arr_colors[3]:
            return random.choice(COLORS)

        idx = arr_colors.index(max(arr_colors))

        return COLORS[idx]

    # to get all the legal moves that the player can make
    def legal_moves(self, hand):
        top = self.top_card()
        moves = []

        if top is None:
            for idx, (t, c, v) in enumerate(hand):
                if t == "W":
                    for col in COLORS:
                        moves.append(("play", idx, col))
                else:
                    moves.append(("play", idx, None))
            if not moves:
                moves.append(("draw", None, None))
            return moves

        for idx, card in enumerate(hand):
            if is_playable(card, top, self.wild_card_color):
                t, c, v = card
                if t == "W":
                    for col in COLORS:
                        moves.append(("play", idx, col))
                else:
                    moves.append(("play", idx, None))

        moves.append(("draw", None, None))
        return moves

    # to apply a move so that it can change the game state
    def apply_move(self, move):
        if self.winner is not None:
            return self.copy()

        action, idx, color_choice = move
        s = self.copy()
        player = s.current_player
        opponent = s.next_player(player)

        if action == "draw":
            s.draw_card(player)
            if s.players_decks[player]:
                drawn_idx = len(s.players_decks[player]) - 1
                t, c, v = s.players_decks[player][drawn_idx]
                if t == "W":
                    choosing_color = True
                    pending_wild_play_index = drawn_idx
                elif is_playable((t, c, v), s.top_card(), s.wild_card_color):
                    s = s.apply_move(("play", drawn_idx, None))
            s.current_player = opponent
            return s

        hand = s.players_decks[player]
        if idx is None or idx < 0 or idx >= len(hand):
            s.draw_card(player)
            s.current_player = opponent
            return s

        keep_turn = False
        card = hand.pop(idx)
        s.played_deck.append(card)
        t, c, v = card

        if t == "W":
            chosen_color = color_choice if color_choice in COLORS else self.colored_cards()
            if v == "Wild4":
                for _ in range(4):
                    s.draw_card(opponent)
                keep_turn = True  # 2-player semantics kept same as before
            s.wild_card_color = chosen_color
        elif t == "A":
            if v == "Draw2":
                for _ in range(2):
                    s.draw_card(opponent)
            s.wild_card_color = None
            keep_turn = True
        else:
            s.wild_card_color = None

        if not keep_turn:
            s.current_player = opponent

        if len(s.players_decks[player]) == 0:
            s.winner = player

        return s

# heuristic 1: to count how many of the agent's cards have the same color as the top card from the played deck
def same_color_cards(state):
    top_card = state.top_card()
    ai_cards = 0

    if state.winner is not None:
        return 100 if state.winner == 1 else -100

    if len(state.players_decks[1]) == 0:
        return 100

    for c in state.players_decks[1]:
        if c[1] == top_card[1]:
            ai_cards += 1

    ai_cards = int((ai_cards / len(state.players_decks[1])) * 100)
    return ai_cards

# heuristic 2: to count how many of the agent's cards are special (Wild4, Draw, Reverse, Skip)
def high_damage_cards(state):
    if state.winner is not None:
        return 10000 if state.winner == 1 else -10000

    score = 0

    score += (len(state.players_decks[0]) - len(state.players_decks[1])) * 25

    action_cards = 0
    wild4_count = 0
    for card in state.players_decks[1]:
        if card[2] in ["Skip", "Reverse", "Draw2"]:
            action_cards += 2
        elif card[2] == "Wild4":
            wild4_count += 1
            action_cards += 3

    score += action_cards * 4

    if wild4_count > 0 and len(state.players_decks[0]) <= 2:
        score -= 15 * wild4_count

    if state.wild_card_color and state.current_player == 1:
        score += 10

    return score

# heuristic 3: combination of the first and second combination
def same_color_and_high_damage_cards(state):
    if state.winner is not None:
        return 10000 if state.winner == 1 else -10000

    score = 0

    score += (len(state.players_decks[0]) - len(state.players_decks[1])) * 25

    action_cards = 0
    wild4_count = 0
    for card in state.players_decks[1]:
        if card[2] in ["Skip", "Reverse", "Draw2"]:
            action_cards += 2
        elif card[2] == "Wild4":
            wild4_count += 1
            action_cards += 3

    score += action_cards * 4

    if wild4_count > 0 and len(state.players_decks[0]) <= 2:
        score -= 15 * wild4_count

    if state.wild_card_color and state.current_player == 1:
        score += 10

    top_card = state.top_card()
    if top_card:
        matching_color = sum(1 for card in state.players_decks[1]
                             if card[1] == top_card[1] or card[0] == "W")
        score += matching_color * 3

    return score

# heuristic 4: returns the difference of number of cards from the human and the ai agent
def cards_length_points(state):
    if state.winner is not None:
        return 10000 if state.winner == 1 else -10000

    human_len = len(state.players_decks[0])
    ai_len = len(state.players_decks[1])
    score = (human_len - ai_len) * 15
    return score

# heuristic 5: returns the difference of playable moves from the human and the ai agent
def playable_cards_points(state):
    if state.winner is not None:
        return 10_000 if state.winner == 1 else -10_000

    human_plays = sum(1 for m in state.legal_moves(state.players_decks[0]) if m[0] == "play")
    ai_plays = sum(1 for m in state.legal_moves(state.players_decks[1]) if m[0] == "play")
    score = (ai_plays - human_plays) * 2
    return score

# heuristic 6: combination of the 4th and 6th heuristic
def cards_length_playable_cards_points(state):
    if state.winner is not None:
        return 10_000 if state.winner == 1 else -10_000

    human_len = len(state.players_decks[0])
    ai_len = len(state.players_decks[1])
    score = (human_len - ai_len) * 15

    human_plays = sum(1 for m in state.legal_moves(state.players_decks[0]) if m[0] == "play")
    ai_plays = sum(1 for m in state.legal_moves(state.players_decks[1]) if m[0] == "play")
    score += (ai_plays - human_plays) * 2

    return score

# to find all the moves that the opponent can make
def find_all_possible_opponent_moves(state):
    all_cards = build_deck()
    used_cards = state.players_decks[1] + state.played_deck

    possible_opponent_cards = []
    for card in all_cards:
        if card in used_cards:
            used_cards.remove(card)
        else:
            possible_opponent_cards.append(card)

    return possible_opponent_cards

# implementation od the minimax algorithm with prunning
def minimax(state, depth, maximizing, alpha=-10 ** 9, beta=10 ** 9):
    if depth == 0 or state.winner is not None:
        return same_color_and_high_damage_cards(state), None

    moves = state.legal_moves(state.players_decks[1]) if maximizing else state.legal_moves(
        state.possible_opponent_cards)
    best_move = None

    if maximizing:
        best_val = -10 ** 9
        for move in moves:
            val, _ = minimax(state.apply_move(move), depth - 1, False, alpha, beta)
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = 10 ** 9
        for move in moves:
            val, _ = minimax(state.apply_move(move), depth - 1, True, alpha, beta)
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best_val, best_move

# to calculate the combinations used for the calculation of hypergeometric probability
def combinations(n, k):
    if k < 0 or k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

# to calculate the hypergeometric probabilty used for expecimax
def hypergeometric_probability(N, K, n, k):
    if k > K or k > n or (n - k) > (N - K):
        return 0
    return combinations(K, k) * combinations(N - K, n - k) / combinations(N, n)

# to calculate all the probabilities used for expectimax
def probabilities_for_expectimax(state):
    seen_cards = defaultdict(int)

    # to count cards in AI's hand
    for card in state.players_decks[1]:
        seen_cards[card] += 1

    # to count cards in discard pile
    for card in state.played_deck:
        seen_cards[card] += 1

    # to get total number of each card type in a full deck
    full_deck_counts = defaultdict(int)
    for color in COLORS:
        full_deck_counts[("N", color, "0")] = 1
        for val in range(1, 10):
            full_deck_counts[("N", color, str(val))] = 2
        for val in ["Skip", "Reverse", "Draw2"]:
            full_deck_counts[("A", color, val)] = 2
    for _ in range(4):
        full_deck_counts[("W", None, "Wild")] = 4
        full_deck_counts[("W", None, "Wild4")] = 4

    # to calculate the remaining cards
    remaining_cards = {}
    for card, total_count in full_deck_counts.items():
        seen_count = seen_cards.get(card, 0)
        remaining_cards[card] = total_count - seen_count

    # total unknown cards = deck + opponent's hand
    opponent_hand_size = len(state.players_decks[0])
    deck_size = len(state.draw_deck)
    total_unknown_cards = opponent_hand_size + deck_size

    # to calculate probability that opponent has each card
    card_probabilities = {}
    for card, remaining_count in remaining_cards.items():
        if remaining_count <= 0:
            card_probabilities[card] = 0
            continue

        # probability that the opponent has at least one of this card
        prob_has_card = 1 - hypergeometric_probability(
            total_unknown_cards,  # total unknown cards
            remaining_count,  # how many of this card are unknown
            opponent_hand_size,  # how many cards opponent has
            0  # probability of having 0 of this card
        )
        card_probabilities[card] = prob_has_card

    return card_probabilities

# implementation of the expectimax algorithm
def expectimax(state, depth, maximizing):
    if depth == 0 or state.winner is not None:
        return same_color_and_high_damage_cards(state), None

    if maximizing:
        moves = state.legal_moves(state.players_decks[1])
        best_val = -float("inf")
        best_move = None

        for move in moves:
            next_state = state.apply_move(move)
            val, _ = expectimax(next_state, depth - 1, False)
            if val > best_val:
                best_val = val
                best_move = move

        return best_val, best_move
    else:
        card_probs = probabilities_for_expectimax(state)
        expected_val = 0
        total_prob = 0

        opponent_hand_size = len(state.players_decks[0])

        likely_cards = [card for card, prob in card_probs.items() if prob > 0.1]

        if not likely_cards:
            next_state = state.apply_move(("draw", None, None))
            val, _ = expectimax(next_state, depth - 1, True)
            return val, None

        mock_hand = likely_cards[:min(opponent_hand_size, len(likely_cards))]
        moves = state.legal_moves(mock_hand)

        if not moves:
            next_state = state.apply_move(("draw", None, None))
            val, _ = expectimax(next_state, depth - 1, True)
            return val, None

        for move in moves:
            if move[0] == "play":
                card_idx = move[1]
                if card_idx < len(mock_hand):
                    card = mock_hand[card_idx]
                    move_prob = card_probs.get(card, 0)
                else:
                    move_prob = 0
            else:
                playable_prob = 1
                for card in mock_hand:
                    if is_playable(card, state.top_card(), state.wild_card_color):
                        playable_prob *= (1 - card_probs.get(card, 0))
                move_prob = playable_prob

            next_state = state.apply_move(move)
            val, _ = expectimax(next_state, depth - 1, True)
            expected_val += val * move_prob
            total_prob += move_prob

        if total_prob > 0:
            return expected_val / total_prob, None
        else:
            return 0, None


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("UNO - Human vs AI (Minimax & Expectimax)")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("arial", 20, bold=True)
BIG = pygame.font.SysFont("arial", 36, bold=True)

WHITE = (240, 240, 240)
BLACK = (10, 10, 10)

# to draw the card shape used for display
def draw_card(x, y, card, face_up=True, wild_card_color=None):
    rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)

    if not face_up:
        card_img = pygame.image.load("images/back.png").convert_alpha()
        card_img = pygame.transform.smoothscale(card_img, (CARD_WIDTH, CARD_HEIGHT))
        screen.blit(card_img, (x, y))
        return rect

    t, c, v = card
    if t == "W":
        if v == "Wild":
            if wild_card_color is None:
                card_img = pygame.image.load("images/Wild.png")
            else:
                card_img = pygame.image.load(f"images/Wild_{wild_card_color}.png")
        else:  # Wild4
            if wild_card_color is None:
                card_img = pygame.image.load("images/Wild4.png")
            else:
                card_img = pygame.image.load(f"images/Wild4_{wild_card_color}.png")
    else:
        card_img = pygame.image.load(f"images/{c}_{v}.png")

    card_img = pygame.transform.smoothscale(card_img, (CARD_WIDTH, CARD_HEIGHT))
    screen.blit(card_img, (x, y))
    return rect

# to get spacings for the cards
def cards_spacing_x_axis(cards, scroll_offset=0):
    n = len(cards)
    if n == 0:
        return []

    # to calculate total width needed for all cards
    total_width = CARD_WIDTH * n + CARD_SPACING * (n - 1)

    # if total width is less than max, center then
    if total_width <= MAX_HAND_WIDTH:
        x = (SCREEN_WIDTH - total_width) // 2
    else:
        # else, start from left with scroll offset
        x = 20 - scroll_offset

    xs = []
    for _ in range(n):
        xs.append(x)
        x += CARD_WIDTH + CARD_SPACING
    return xs

# to draw the scene for the game
def draw_scene(state: Game, hover_idx=None, color_picker=False, scroll_offset=0):
    screen.fill((0, 0, 0))

    deck_rect = pygame.Rect(CENTER_X - 220, CENTER_Y - CARD_HEIGHT // 2, CARD_WIDTH, CARD_HEIGHT)
    deck_img = pygame.image.load("images/back.png").convert_alpha()
    deck_img = pygame.transform.smoothscale(deck_img, (CARD_WIDTH, CARD_HEIGHT))
    screen.blit(deck_img, deck_rect.topleft)
    dcount = FONT.render(f"Deck: {len(state.draw_deck)}", True, (255, 255, 255))
    screen.blit(dcount, (deck_rect.x, deck_rect.y + CARD_HEIGHT + 8))

    discard_rect = pygame.Rect(CENTER_X - CARD_WIDTH // 2, CENTER_Y - CARD_HEIGHT // 2, CARD_WIDTH, CARD_HEIGHT)
    top = state.top_card()
    if top:
        draw_card(discard_rect.x, discard_rect.y, top, face_up=True, wild_card_color=state.wild_card_color)
    else:
        pygame.draw.rect(screen, (160, 160, 160), discard_rect, border_radius=12)
    screen.blit(FONT.render("Discard", True, (255, 255, 255)), (discard_rect.x - 6, discard_rect.y + CARD_HEIGHT + 8))

    # to use scroll_offset for human hand
    human_xs = cards_spacing_x_axis(state.players_decks[0], scroll_offset)
    for idx, x in enumerate(human_xs):
        # to only draw cards that are visible on screen
        if x + CARD_WIDTH > 0 and x < SCREEN_WIDTH:
            draw_card(
                x,
                HAND_Y_HUMAN,
                state.players_decks[0][idx],
                face_up=True,
                wild_card_color=None
            )
            if hover_idx is not None and idx == hover_idx:
                pygame.draw.rect(screen, (255, 255, 255), (x - 2, HAND_Y_HUMAN - 2, CARD_WIDTH + 4, CARD_HEIGHT + 4), 2,
                                 border_radius=12)

    if human_xs:
        screen.blit(FONT.render(f"Your cards: {len(human_xs)}", True, (255, 255, 255)), (human_xs[0], SCREEN_HEIGHT - 30))
        # to show scroll indicator if hand is too wide
        total_width = CARD_WIDTH * len(human_xs) + CARD_SPACING * (len(human_xs) - 1)

    ai_xs = cards_spacing_x_axis(state.players_decks[1])
    for idx, x in enumerate(ai_xs):
        draw_card(x, HAND_Y_AI, state.players_decks[1][idx], face_up=False, wild_card_color=None)
    if ai_xs:
        screen.blit(FONT.render(f"AI cards: {len(ai_xs)}", True, (255, 255, 255)),
                    (ai_xs[0], HAND_Y_AI + CARD_HEIGHT + 8))

    turn_txt = "Your turn" if state.current_player == 0 else "AI is thinking"
    tint = BIG.render(turn_txt, True, (255, 255, 255))
    screen.blit(tint, (20, HAND_Y_HUMAN - 44))

    color_palette = [(237, 28, 36), (0, 166, 81), (0, 149, 218), (255, 222, 0)]
    color_names = ["Red", "Green", "Blue", "Yellow"]

    picker_rects = []
    if color_picker:
        prompt = BIG.render("Choose color for WILD:", True, (255, 255, 255))
        screen.blit(prompt, (CENTER_X - prompt.get_width() // 2, CENTER_Y - 120))
        color_palette = [(237, 28, 36), (0, 166, 81), (0, 149, 218), (255, 222, 0)]
        color_names = ["Red", "Green", "Blue", "Yellow"]
        for idx in range(4):
            r = pygame.Rect(CENTER_X - 220 + idx * 120, CENTER_Y - 40, 100, 100)
            pygame.draw.rect(screen, color_palette[idx], r, border_radius=12)
            pygame.draw.rect(screen, (0, 0, 0), r, border_radius=12, width=1)
            name = FONT.render(color_names[idx], True, (0, 0, 0))
            if idx == 0:
                dx = r.x + 33
            elif idx == 1 or idx == 3:
                dx = r.x + 25
            else:
                dx = r.x + 30
            screen.blit(name, (dx, r.y + r.height // 2 - 10))
            picker_rects.append((r, COLORS[idx]))

    return deck_rect, discard_rect, human_xs, picker_rects

# to start the game with already made test cases
def start_game_test_cases():
    hands, discard, deck = get_test_case(1)
    possible_cards = []
    return Game(draw_deck=deck, played_deck=discard, players_decks=hands, current_player=0,
                possible_opponent_cards=possible_cards)

# to start the game with random shuffle
def start_game():
    deck = build_deck()
    discard = []

    while deck:
        card = deck.pop()
        if card[0] == "N":
            discard.append(card)
            break
        else:
            deck.insert(0, card)

    hands = [[], []]
    for _ in range(7):
        hands[0].append(deck.pop())
        hands[1].append(deck.pop())

    possible_cards = []

    return Game(draw_deck=deck, played_deck=discard, players_decks=hands, current_player=0,
                possible_opponent_cards=possible_cards)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("UNO - Human vs AI (Minimax & Expectimax)")
    clock = pygame.time.Clock()
    FONT = pygame.font.SysFont("arial", 20, bold=True)
    BIG = pygame.font.SysFont("arial", 36, bold=True)

    rounds = 0
    total_time = 0

    state = start_game()
    running = True
    hover_idx = None
    choosing_color = False
    pending_wild_play_index = None
    scroll_offset = 0
    max_scroll_offset = 0
    mouse_dragging = False
    drag_start_x = 0
    drag_start_offset = 0

    ai_timer_ms = 0

    while running:
        dt = clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        hover_idx = None

        # to calculate maximum scroll offset based on current hand size
        total_hand_width = len(state.players_decks[0]) * CARD_WIDTH + (len(state.players_decks[0]) - 1) * CARD_SPACING
        max_scroll_offset = max(0, total_hand_width - MAX_HAND_WIDTH + 40)  # 40px padding

        if state.winner is not None:
            total_time /= rounds if rounds > 0 else 1
            winner_txt = "You win!" if state.winner == 0 else "AI agent wins!"
            overlay = BIG.render(winner_txt, True, (255, 255, 255))
            screen.fill((0, 0, 0))
            screen.blit(overlay, (CENTER_X - overlay.get_width() // 2, CENTER_Y - 30))
            again = FONT.render("Press R to restart or ESC to quit.", True, (255, 255, 255))
            screen.blit(again, (CENTER_X - again.get_width() // 2, CENTER_Y + 20))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit(0)
                    if event.key == pygame.K_r:
                        state = start_game()
                        scroll_offset = 0  # to reset scroll when restarting
            continue

        if state.current_player == 0 and not choosing_color:
            human_xs = cards_spacing_x_axis(state.players_decks[0], scroll_offset)
            for i, x in enumerate(human_xs):
                # to only check cards that are visible on screen
                if x + CARD_WIDTH > 0 and x < SCREEN_WIDTH:
                    rect = pygame.Rect(x, HAND_Y_HUMAN, CARD_WIDTH, CARD_HEIGHT)
                    if rect.collidepoint(mouse_pos):
                        hover_idx = i
                        break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = start_game()
                    scroll_offset = 0  # to reset scroll when restarting
                if event.key == pygame.K_ESCAPE:
                    running = False

                # to handle scrolling with arrow keys
                if event.key == pygame.K_LEFT:
                    scroll_offset = max(0, scroll_offset - SCROLL_SPEED)
                if event.key == pygame.K_RIGHT:
                    scroll_offset = min(max_scroll_offset, scroll_offset + SCROLL_SPEED)

            # Handle mouse wheel scrolling
            if event.type == pygame.MOUSEWHEEL:
                if event.x < 0:  # to scroll up/left
                    scroll_offset = max(0, scroll_offset - int(SCROLL_SPEED * MOUSE_SCROLL_SENSITIVITY))
                elif event.x > 0:  # to scroll down/right
                    scroll_offset = min(max_scroll_offset, scroll_offset + int(SCROLL_SPEED * MOUSE_SCROLL_SENSITIVITY))

            # to handle mouse drag scrolling
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # to check if clicking in the hand area but not on a card
                human_xs = cards_spacing_x_axis(state.players_decks[0], scroll_offset)
                clicked_on_card = False
                for i, x in enumerate(human_xs):
                    if x + CARD_WIDTH > 0 and x < SCREEN_WIDTH:
                        rect = pygame.Rect(x, HAND_Y_HUMAN, CARD_WIDTH, CARD_HEIGHT)
                        if rect.collidepoint(event.pos):
                            clicked_on_card = True
                            break

                if not clicked_on_card and HAND_Y_HUMAN <= event.pos[1] <= HAND_Y_HUMAN + CARD_HEIGHT:
                    mouse_dragging = True
                    drag_start_x = event.pos[0]
                    drag_start_offset = scroll_offset

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_dragging = False

            if event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx = drag_start_x - event.pos[0]
                scroll_offset = max(0, min(max_scroll_offset, drag_start_offset + dx))

            if state.current_player == 0:
                if choosing_color:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        _, _, _, picker_rects = draw_scene(state, hover_idx, True, scroll_offset)
                        for rect, col in picker_rects:
                            if rect.collidepoint(event.pos):
                                if pending_wild_play_index is not None:
                                    state = state.apply_move(("play", pending_wild_play_index, col))
                                choosing_color = False
                                pending_wild_play_index = None
                else:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        deck_rect, _, hxs, _ = draw_scene(state, hover_idx, False, scroll_offset)
                        clicked = False

                        for i, x in enumerate(hxs):
                            # to only check cards that are visible on screen
                            if x + CARD_WIDTH > 0 and x < SCREEN_WIDTH:
                                rect = pygame.Rect(x, HAND_Y_HUMAN, CARD_WIDTH, CARD_HEIGHT)
                                if rect.collidepoint(event.pos):
                                    if i < len(state.players_decks[0]):
                                        hand = state.players_decks[0]
                                        t, c, v = hand[i]

                                        legal = state.legal_moves(state.players_decks[0])
                                        if t == "W":
                                            can_play = any(mv[0] == "play" and mv[1] == i for mv in legal)
                                            if can_play:
                                                choosing_color = True
                                                pending_wild_play_index = i
                                        else:
                                            can_play = any(mv[0] == "play" and mv[1] == i for mv in legal)
                                            if can_play:
                                                state = state.apply_move(("play", i, None))
                                    clicked = True
                                    break

                        if not clicked and deck_rect.collidepoint(event.pos):
                            state = state.apply_move(("draw", None, None))

        if state.current_player == 1 and state.winner is None:
            ai_timer_ms += dt
            if ai_timer_ms >= AI_THINK_DELAY_MS:
                ai_timer_ms = 0
                possible_cards = find_all_possible_opponent_moves(state)
                state.possible_opponent_cards = possible_cards
                start_time = time.time()
                if AI_ALGORITHM == "expectimax":
                    _, best = expectimax(state, SEARCH_DEPTH, True)
                elif AI_ALGORITHM == "minimax":
                    _, best = minimax(state, SEARCH_DEPTH, True)
                end_time = time.time()
                total_time += (end_time-start_time)
                rounds += 1
                if best is None:
                    best = ("draw", None, None)
                state = state.apply_move(best)

        draw_scene(state, hover_idx, choosing_color, scroll_offset)
        pygame.display.flip()

    pygame.quit()