_two_in_lines = np.array([[1, 1, 0, 0],
                          [1, 0, 1, 0],
                          [1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 1]])

_three_in_lines = np.array([[1, 1, 1, 0],
                            [0, 1, 1, 1]])

_arbitrary_lines = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

_win_line = np.array([1, 1, 1, 1])

def get_left_diagonal_lines(state):
    left_diagonal_lines = []
    for d_index in range(-5, 7):
        left_diagonal_lines.append(state.diagonal(d_index))
    return left_diagonal_lines


def get_right_diagonal_lines(state):
    return get_left_diagonal_lines(np.fliplr(state))  # anti-diagonals


def get_vertical_lines(state):
    vertical_lines = []
    for c in range(7):
        vertical_lines.append(state[:, c])
    return vertical_lines


def get_horizontal_lines(state):
    horizontal_lines = []
    for r in range(6):
        horizontal_lines.append(state[r, :])
    return horizontal_lines


def count_subarray(array, subarray):
    count = 0
    if array.size >= subarray.size:
        for i in range(array.size - subarray.size + 1):
            _array = array[i:i + subarray.size]
            if (_array == subarray).all():
                count += 1
    return count


# print("found", np.equal([5, -1], _no_diagonal).all(axis=1).any())


def win_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines):
    count = 0
    check_line = np.multiply(_win_line, marker)
    for line in vertical_lines:
        count += count_subarray(line, check_line)
    for line in horizontal_lines:
        count += count_subarray(line, check_line)
    for line in left_diagonal_lines:
        count += count_subarray(line, check_line)
    for line in right_diagonal_lines:
        count += count_subarray(line, check_line)
    return count * 1000000


def two_in_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines):
    count = 0
    for two_in_line in _two_in_lines:
        check_line = np.multiply(two_in_line, marker)
        for line in vertical_lines:
            count += count_subarray(line, check_line)
        for line in horizontal_lines:
            count += count_subarray(line, check_line)
        for line in left_diagonal_lines:
            count += count_subarray(line, check_line)
        for line in right_diagonal_lines:
            count += count_subarray(line, check_line)
    return count * 50


def three_in_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines):
    count = 0
    for three_in_line in _three_in_lines:
        check_line = np.multiply(three_in_line, marker)
        for line in vertical_lines:
            count += count_subarray(line, check_line)
        for line in horizontal_lines:
            count += count_subarray(line, check_line)
        for line in left_diagonal_lines:
            count += count_subarray(line, check_line)
        for line in right_diagonal_lines:
            count += count_subarray(line, check_line)
    return count * 100


def arbitrary_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines):
    count = 0
    for arbitrary_line in _arbitrary_lines:
        check_line = np.multiply(arbitrary_line, marker)
        for line in vertical_lines:
            count += count_subarray(line, check_line)
        for line in horizontal_lines:
            count += count_subarray(line, check_line)
        for line in left_diagonal_lines:
            count += count_subarray(line, check_line)
        for line in right_diagonal_lines:
            count += count_subarray(line, check_line)
    return count * 10

def score(state):
    vertical_lines = get_vertical_lines(state)  # get the vertical line containing the piece
    horizontal_lines = get_horizontal_lines(state)  # get the horizontal line containing the piece
    left_diagonal_lines = get_left_diagonal_lines(state)
    right_diagonal_lines = get_right_diagonal_lines(state)
    value = 0
    for marker in [1, -1]:
        # value += win_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines)
        value += _score(state, marker)
        # value += three_in_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines)
        # value += two_in_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines)
        # value += arbitrary_lines(marker, vertical_lines, horizontal_lines, left_diagonal_lines, right_diagonal_lines)

    value += 138 + np.multiply(_eval_table, state).sum()
    return value