# Implementation of the game UNO using the Minimax and Expectimax algorithms
This project was made for the course Artificial Intelligence at the FCSE - Skopje. \
In this project I implemented and analyzed the UNO card game in both single-agent (human vs. AI) and multi-agent (AI vs. AI) environments. \
My main focus was to develop an intelligent agent that makes strategic decisions using Minimax and Expectimax algorithms. \
The agent evaluates possible game states and selects optimal moves to maximize winning chances. \
To handle the complexity and uncertainty of UNO, the implementation includes:
- Search depths of 3, 4 and 5
- Six heuristics that estimate game state quality based on: the number of playable cards, cards diversity, number of special (high damage) cards, card count difference between players and combinations of these features 

In total, 36 models were tested (2 algorithms × 3 depths × 6 heuristics).

## Game analysis
Minimax assumes a perfectly rational opponent and works well for short-term strategies. On the other hand, Expectimax incorporates probabilities (using the hypergeometric distribution) to model uncertainity, especially regarding the opponent's hidden cards and the draw pile. \
The performance of both algorithms was tested on 9 predefined test cases, covering scenarios with only numbered cards, only special cards, or a mix of both. This ensured fair evaluation across all 36 models. \
The results showed that heuristics and search depth heavily influence decision quality and execution time. \
After testing all the 36 models in the Human vs. AI scenario, I've come to conclusion that the best models for the Minimax algorith were the ones where I used the same_color_and_high_damage_cards heuristic with depths 3 or 4. On the other hand, the best models for the Expectimax algorithm were also the ones where I used the same_color_and_high_damage_cards heuristic with depths 3, 4 or 5.\
For testing the performances of the game with two agents, I used the Minimax and Expectimax algorithms with heuristic same_color_and_high_damage_cards with depth 4 in four different scenarios (Minimax vs. Minimax, Minimax vs. Expectimax, Expectimax vs. Minimax and Expectimax vs. Expectimax). The results showed that the Minimax algorithm was slightly better than the Expectimax algorithm.

## Requirements
To run the project you need:
- Python 3.8 or higher
- Required libraries:
    - pygame
    - random
    - math
    - sys
    - time
    - collections

## Dependencies
Install via pip: 
```
pip install pygame
