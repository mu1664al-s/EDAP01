import time

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


player_mode = False


def opponents_move(env):
    env.change_player()  # change to opponent
    if not player_mode:
        avmoves = env.available_moves()
        if not avmoves:
            env.change_player()  # change back to student before returning
            return -1

        # TODO: Optional? change this to select actions with your policy too
        # that way you get way more interesting games, and you can see if starting
        # is enough to guarantee a win
        action = random.choice(list(avmoves))
    else:
        action = int(input("Move: "))

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

_three_line = np.array([0, 1, 1, 1, 0])

_two_line = np.array([[0, 1, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [0, 0, 1, 1, 0]])


def available_moves(state):
    _avmoves = []
    for i in range(7):  # for each move
        col_i = state[:, i]  # get the column
        empty_slots = np.where(col_i == 0)[0]  # indices of empty slots
        # print("empty slots", empty_slots)
        if empty_slots.size > 0:
            _avmoves.append((empty_slots[-1], i))  # pick the lowest one
    return _avmoves


def compare_to_arrays(array, arrays):
    for _array in arrays:
        if np.array_equal(_array, array):
            return True
    return False


def filled_below(bellow_sub_array):
    return np.count_nonzero(bellow_sub_array == 0) == 0


def check_four(win, value):
    marker = -1 if value < 0 else 1
    if abs(value) == 4:
        return win * marker
    return 0


def check_five(sub_array, _filled_bellow, three, two):
    value = 0
    marker = -1 if sum(sub_array) < 0 else 1
    if _filled_bellow and np.array_equal(sub_array, np.multiply(_three_line, marker)):
        value = three * marker
    if _filled_bellow and compare_to_arrays(sub_array, np.multiply(_two_line, marker)):
        value = two * marker
    return value


def check_straight_line(state, win, three, two):
    line_score = 0
    rows, columns = state.shape
    for i in range(rows):
        for j in range(columns - 3):
            value = state[i][j:j + 4].sum()
            four_value = check_four(win, value)
            if four_value < 0:
                return four_value
            line_score += four_value
            if rows == 6 and j + 5 < columns:     # if checking for rows
                sub_array = state[i][j:j + 5]
                _filled_bellow = True
                if i + 1 < rows:
                    _filled_bellow = filled_below(state[i + 1][j:j + 5])
                line_score += check_five(sub_array, _filled_bellow, three, two)
    return line_score


def check_diagonal_line(state, win, three, two):
    line_score = 0
    rows, columns = state.shape
    for i in range(rows - 3):
        for j in range(columns - 3):
            value = 0
            sub_array = []
            for k in range(4):
                value += state[i + k][j + k]
                sub_array.append(state[i + k][j + k])
            #   Check 4 in row
            four_value = check_four(win, value)
            if four_value < 0:
                return four_value
            line_score += four_value
            # Check the rest
            if i <= 1 and j <= 2:
                sub_array.append(state[i + 4][j + 4])
                _filled_bellow = True
                if i + 1 <= 2:
                    bellow_array = []
                    for k in range(4):
                        bellow_array.append(state[i + 1 + k][j + k])
                    if i + 1 <= 1:
                        bellow_array.append(state[i + 1 + 4][j + 4])
                    _filled_bellow = filled_below(bellow_array)
                line_score += check_five(sub_array, _filled_bellow, three, two)
    return line_score


def _score(state):
    node_score = 0
    three = 100000
    two = 10000
    win = 1000000
    # Test rows
    node_score += check_straight_line(state, win, three, two)

    # Test columns on transpose array
    reversed_board = np.matrix.transpose(state)
    node_score += check_straight_line(reversed_board, win, three, two)

    # Test diagonal
    check_diagonal_line(state, win, three, two)

    reversed_board = np.fliplr(state)
    # Test reverse diagonal
    check_diagonal_line(reversed_board, win, three, two)

    return node_score


def score(state):
    return _score(state) + 138 + np.multiply(_eval_table, state).sum()


def alhabeta(node, depth, alpha, beta, maximizing_player):
    _move = 3
    _alpha = alpha
    _beta = beta
    if depth == 0:
        return score(node), _move
    if maximizing_player:
        value = -1000000000
        for move in available_moves(node):
            r, c = move
            node[r][c] = 1  # make move
            child_score = np.maximum(value, alhabeta(node, depth - 1, alpha, beta, False)[0])
            node[r][c] = 0  # reverse move (more efficient then copying the matrix)
            _alpha = np.maximum(_alpha, child_score)
            if value < child_score:
                value = child_score
                _move = c
            if _alpha >= _beta:
                break  # beta cutoff
        return value, _move
    else:
        value = 1000000000
        for move in available_moves(node):
            r, c = move
            node[r][c] = -1  # make move
            value = np.minimum(value, alhabeta(node, depth - 1, alpha, beta, True)[0])
            node[r][c] = 0  # reverse move (more efficient then copying the matrix)
            _beta = np.minimum(_beta, value)
            """if value > child_score:
                value = child_score
                _move = c"""
            if _beta <= _alpha:
                # print("value: ", value, "move: ", c, "alpha: ", alpha, "beta: ", beta)
                break  # alpha cutoff
        return value, _move


longest_move = 0
sum_moves = 0
num_moves = 0


def student_move(state):
    """
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """
    # print(state)
    global longest_move, sum_moves, num_moves
    start_time = time.time()
    value, move = alhabeta(state, 4, -500000, 500000, True)
    print("move", move, "score", value)
    print("____________________________________________________________________________________________________")
    move_time = time.time() - start_time
    sum_moves += move_time
    num_moves += 1
    longest_move = np.maximum(longest_move, move_time)
    print("longest move time in seconds: ", longest_move, "mean move time in seconds: ", sum_moves / num_moves)
    return move


_lose = 0
_win = 0
_draw = 0


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
    global _win, _lose, _draw

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
            print("playing against server")
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
                if player_mode:
                    # Print current game state
                    print(state)
                    print()

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env)
                if player_mode:
                    # Print current game state
                    print(state)
                    print()

        # Check if the game is over
        if result != 0:
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:
                print("You won!")
                _win += 1
            elif result == 0.5:
                print("It's a draw!")
                _draw += 1
            elif result == -1:
                print("You lost!")
                _lose += 1
            elif result == -10:
                print("You made an illegal move and have lost!")
            else:
                print("Unexpected result result={}".format(result))
            if not vs_server:
                print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
        else:
            print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

        if not player_mode:
            # Print current game state
            print(state)
            print()


def main():
    while _win < 1000:
        play_game(vs_server=True)
        print("win", _win, "lose", _lose, "draw", _draw)
        if _lose > 0:
            break
    # TODO: Change vs_server to True when you are ready to play against the server
    # the results of your games there will be logged


if __name__ == "__main__":
    main()
