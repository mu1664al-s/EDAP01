import time
from functools import reduce

import gym
import random
import requests
import numpy as np
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

# SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["mu1664al-s"]  # TODO: fill this list with your stil-id's


def call_server(move):
    res = requests.post(SERVER_ADRESS + "move",
                        data={
                            "stil_id": STIL_ID,
                            "move": move,  # -1 signals the system to start a new game. any running game is counted as
                            # a loss
                            "api_key": API_KEY,
                        })
    # For safety some response checking is done here
    if res.status_code != 200:
        print("Server gave a bad response, error code={}".format(res.status_code))
        exit()
    if not res.json()['status']:
        print("Server returned a bad status. Return message: ")
        print(res.json()['msg'])
        exit()
    return res


"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""


def opponents_move(env):
    env.change_player()  # change to opponent
    avmoves = env.available_moves()
    if not avmoves:
        env.change_player()  # change back to student before returning
        return -1

    # TODO: Optional? change this to select actions with your policy too
    # that way you get way more interesting games, and you can see if starting
    # is enough to guarantee a win
    action = random.choice(list(avmoves))

    state, reward, done, _ = env.step(action)
    if done:
        if reward == 1:  # reward is always in current players view
            reward = -1
    env.change_player()  # change back to student before returning
    return state, reward, done


_eval_table = np.array([[3, 4, 5, 7, 5, 4, 3],
                        [4, 6, 8, 10, 8, 6, 4],
                        [5, 8, 11, 13, 11, 8, 5],
                        [5, 8, 11, 13, 11, 8, 5],
                        [4, 6, 8, 10, 8, 6, 4],
                        [3, 4, 5, 7, 5, 4, 3]])


def available_moves(state):
    _avmoves = []
    for i in range(7):  # for each move
        col_i = state[:, i]  # get the column
        empty_slots = np.where(col_i == 0)[0]  # indices of empty slots
        # print("empty slots", empty_slots)
        if empty_slots.size > 0:
            _avmoves.append((empty_slots[-1], i))  # pick the lowest one
    return _avmoves


def _score(state, marker):
    # Test rows
    node_score = 0
    three = 10
    two = 3
    for i in range(6):
        for j in range(7 - 3):
            value = state[i][j:j + 4].sum()
            if value == 4 * marker:
                node_score += 1000000 * marker
            if value == 3 * marker:
                node_score += three * marker
            if value == 2 * marker:
                node_score += two * marker

    # Test columns on transpose array
    reversed_board = np.matrix.transpose(state)
    for i in range(7):
        for j in range(6 - 3):
            value = reversed_board[i][j:j + 4].sum()
            if value == 4 * marker:
                node_score += 1000000 * marker
            if value == 3 * marker:
                node_score += three * marker
            if value == 2 * marker:
                node_score += two * marker

    # Test diagonal
    for i in range(6 - 3):
        for j in range(7 - 3):
            value = 0
            for k in range(4):
                value += state[i + k][j + k]
                if value == 4 * marker:
                    node_score += 1000000 * marker
                if value == 3 * marker:
                    node_score += three * marker
                if value == 2 * marker:
                    node_score += two * marker

    reversed_board = np.fliplr(state)
    # Test reverse diagonal
    for i in range(6 - 3):
        for j in range(7 - 3):
            value = 0
            for k in range(4):
                value += reversed_board[i + k][j + k]
                if value == 4 * marker:
                    node_score += 1000000 * marker
                if value == 3 * marker:
                    node_score += three * marker
                if value == 2 * marker:
                    node_score += two * marker

    return node_score


def score(state):
    value = 0
    for marker in [1, -1]:
        value += _score(state, marker)

    value += 138 + np.multiply(_eval_table, state).sum()
    return value


def alhabeta(node, depth, alpha, beta, maximizing_player):
    _move = 3
    if depth == 0:
        return score(node), _move
    if maximizing_player:
        value = -np.inf
        for move in available_moves(node):
            r, c = move
            node[r][c] = 1  # make move
            child_score = np.maximum(value, alhabeta(node, depth - 1, alpha, beta, False)[0])
            node[r][c] = 0  # reverse move (more efficient then copying the matrix)
            alpha = np.maximum(alpha, child_score)
            if value < child_score:
                value = child_score
                _move = c
            if alpha >= beta:
                break  # beta cutoff
        return value, _move
    else:
        value = np.inf
        for move in available_moves(node):
            r, c = move
            node[r][c] = -1  # make move
            child_score = np.minimum(value, alhabeta(node, depth - 1, alpha, beta, True)[0])
            node[r][c] = 0  # reverse move (more efficient then copying the matrix)
            beta = np.minimum(beta, child_score)
            if value > child_score:
                value = child_score
                _move = c
            if beta <= alpha:
                break  # alpha cutoff
        return value, _move


longest_move = 0


def student_move(state):
    """
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """
    # print(state)
    global longest_move
    start_time = time.time()
    value, move = alhabeta(state, 5, -np.inf, np.inf, True)
    print("move", move, "score", value)
    print("____________________________________________________________________________________________________")
    longest_move = np.maximum(longest_move, time.time() - start_time)
    print("longest move time in seconds", longest_move)
    return move


lose = 0
win = 0
draw = 0


def play_game(vs_server=False):
    """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """
    global win, lose, draw

    # default state
    state = np.zeros((6, 7), dtype=int)

    # setup new game
    if vs_server:
        # Start a new game
        res = call_server(-1)  # -1 signals the system to start a new game. any running game is counted as a loss

        # This should tell you if you or the bot starts
        print(res.json()['msg'])
        botmove = res.json()['botmove']
        state = np.array(res.json()['state'])
    else:
        # reset game to starting state
        env.reset(board=None)
        # determine first player
        student_gets_move = random.choice([True, False])
        if student_gets_move:
            print('You start!')
            print()
        else:
            print('Bot starts!')
            print()

    # Print current game state
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)
    print()

    done = False
    while not done:

        # make both student and bot/server moves
        if vs_server:
            # Select your move
            stmove = student_move(state)  # TODO: change input here

            # Send your move to server and get response
            res = call_server(stmove)
            print(res.json()['msg'])

            # Extract response values
            result = res.json()['result']
            botmove = res.json()['botmove']
            state = np.array(res.json()['state'])
        else:
            if student_gets_move:
                # Select your move
                stmove = student_move(state)  # TODO: change input here
                # Execute your move
                avmoves = env.available_moves()
                if stmove not in avmoves:
                    print("You tried to make an illegal move! Games ends.")
                    break
                state, result, done, _ = env.step(stmove)

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env)

        # Check if the game is over
        if result != 0:
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:
                print("You won!")
                win += 1
            elif result == 0.5:
                print("It's a draw!")
                draw += 1
            elif result == -1:
                print("You lost!")
                lose += 1
            elif result == -10:
                print("You made an illegal move and have lost!")
            else:
                print("Unexpected result result={}".format(result))
            if not vs_server:
                print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
        else:
            print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

        # Print current game state
        print(state)
        print()


def main():
    while win < 21:
        play_game(vs_server=True)
        print("win", win, "lose", lose, "draw", draw)
    # TODO: Change vs_server to True when you are ready to play against the server
    # the results of your games there will be logged


if __name__ == "__main__":
    main()
