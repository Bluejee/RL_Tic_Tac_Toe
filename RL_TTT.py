"""
This contains the classes and functions that are necessary for the RL for the Tic Tac Toe Game.
"""
import numpy as np
import json


class GameBoard:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros([board_size, board_size])

    def check_win(self):
        """
        This function checks and returns true(win)if there exists any row or column or either of the diagonals has
        all Xs(1s) or Os(-1s).
        :return: Winner or 0
        """
        # check rows and columns
        for i in range(self.board_size):
            if (np.sum(self.board[:, i]) == self.board_size) or (np.sum(self.board[i, :]) == self.board_size):
                return 1
            if (np.sum(self.board[:, i]) == -self.board_size) or (np.sum(self.board[i, :]) == -self.board_size):
                return -1

        # check diagonals.
        if (np.sum(np.diag(self.board)) == self.board_size) or (
                np.sum(np.diag(np.fliplr(self.board))) == self.board_size):
            return 1
        if (np.sum(np.diag(self.board)) == -self.board_size) or (
                np.sum(np.diag(np.fliplr(self.board))) == -self.board_size):
            return -1

        return 0


class Brain:
    def __init__(self, learning_rate, exploration_rate=1, discounted_return=0.9, exploration_decay=0.00001,
                 table_name=''):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.discounted_return = discounted_return
        if table_name == '':
            self.qtable = {}  # key is the game state and value is the q value.
        else:
            self.load_qtable(table_name)
        self.action_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def load_qtable(self, name):
        with open(name + '.json', "r") as f:
            self.qtable = json.load(f)

    def save_qtable(self, name):
        with open(name + '.json', "w") as f:
            json.dump(self.qtable, f)

    def get_action(self, current_state):
        if (np.random.random() <= self.exploration_rate) or current_state not in self.qtable:
            action = np.random.choice(self.action_list)
        else:
            qvalue_list = self.qtable[current_state]
            max_qvalue_index = np.where(qvalue_list == np.max(qvalue_list))
            max_qvalue_index = max_qvalue_index[0]
            action = self.action_list[np.random.choice(max_qvalue_index)]
        return action

    def learn(self, learn_state, next_state, action_taken, reward):
        if learn_state in self.qtable:
            if next_state not in self.qtable:
                self.qtable[next_state] = np.zeros(len(self.action_list))

            learned_value = reward + self.discounted_return * (np.max(self.qtable[next_state]))
            # print(action_taken,np.where(self.action_list == action_taken))
            # print(self.qtable[learn_state])
            self.qtable[learn_state][np.where(self.action_list == action_taken)[0]] = (1 - self.learning_rate) * \
                                                                                      self.qtable[learn_state][
                                                                                          np.where(
                                                                                              self.action_list == action_taken)[
                                                                                              0]] + self.learning_rate * (
                                                                                          learned_value)
        else:
            self.qtable[learn_state] = np.zeros(len(self.action_list))

    def decay_exploration(self):
        if self.exploration_rate >= 0.1:
            self.exploration_rate -= self.exploration_decay


class Agent:
    def __init__(self, player_value, table_name=''):
        self.player_value = player_value
        # self.current_state = tuple(np.append(self.player_value, initial_board_state.flatten()))
        self.current_state = 'Start'
        self.current_action = -1
        # self.previous_state = tuple(np.append(self.player_value, initial_board_state.flatten()))
        self.previous_state = 'Start'
        self.previous_action = -1
        self.brain = Brain(0.02, 1, table_name=table_name)
        self.brain.learn(self.previous_state, self.current_state, self.previous_action, 0)

    def sense_state(self, game_board):
        self.previous_state = self.current_state
        self.current_state = tuple(np.append(self.player_value, game_board.flatten()))

    def make_move(self):
        self.previous_action = self.current_action
        self.current_action = self.brain.get_action(self.current_state)
        # print(self.previous_action,self.current_action)
        return self.current_action

    def new_game(self):
        self.current_state = 'Start'
        self.previous_state = 'Start'
        self.current_action = -1
        self.previous_action = -1

    def set_reward_learn(self, reward):
        self.brain.learn(self.previous_state, self.current_state, self.previous_action, reward)


def lets_2_have_fun():
    a = GameBoard()
    player = 1
    while a.check_win() == 0 and np.isin(0, a.board):
        print(a.board)
        print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        play = int(input(f"Enter play position for Player {player} :: "))
        a.board[(play - 1) // 3, (play - 1) % 3] = player

        if a.check_win() != 0:
            print(f"Player {a.check_win()} has Won!")
        else:
            player *= -1
    else:
        if a.check_win() == 0:
            print("Draw!")


def lets_1_and_random():
    a = GameBoard()
    player = 1
    while a.check_win() == 0 and np.isin(0, a.board):
        print(a.board)
        print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        if player == 1:
            play = int(input(f"Enter play position for Player {player} :: "))
            a.board[(play - 1) // 3, (play - 1) % 3] = player
        else:
            while True:
                comp_play = np.random.randint(1, 10)
                if a.board[(comp_play - 1) // 3, (comp_play - 1) % 3] == 0:
                    a.board[(comp_play - 1) // 3, (comp_play - 1) % 3] = player
                    break
                else:
                    pass

        if a.check_win() != 0:
            print(f"Player {a.check_win()} has Won!")
        else:
            player *= -1

    else:
        if a.check_win() == 0:
            print("Draw!")


def lets_1_and_ai(ai: Agent):
    ai = ai
    replay = 'y'
    while replay == 'y':
        # Let the human make the first move.
        player = 1
        reward = 0
        game_env = GameBoard(3)
        ai.new_game()
        while True:
            if player == 1:
                print(game_env.board)
                print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
                play = int(input(f"Enter play position for Player {player} :: "))
                game_env.board[(play - 1) // 3, (play - 1) % 3] = player
                player *= -1
            else:
                # Sense
                ai.sense_state(game_env.board)
                # Learn
                if game_env.check_win() == 1:
                    print(game_env.board)
                    print('You Defeated the AI!')
                    reward = -1
                    ai.set_reward_learn(reward)
                    break
                elif game_env.check_win() == -1:
                    print(game_env.board)
                    print('You Lost to the AI!')
                    reward = 1
                    ai.set_reward_learn(reward)
                    break
                else:
                    if not np.isin(0, game_env.board):
                        print(game_env.board)
                        print('Its a Draw!')
                        ai.set_reward_learn(reward)
                        break
                    ai.set_reward_learn(reward)
                # Act
                while True:
                    play = ai.make_move()
                    if game_env.board[(play - 1) // 3, (play - 1) % 3] == 0:
                        break
                    else:
                        reward = -2
                        ai.sense_state(np.array([-2]))
                        ai.set_reward_learn(reward)
                        ai.sense_state(game_env.board)
                game_env.board[(play - 1) // 3, (play - 1) % 3] = player
                player *= -1

        replay = input('do you want to play again?(y/n)')


def lets_ai_and_ai(n_games, ai1: Agent, ai2: Agent, d):
    n = 1
    score = {'ai1': 0, 'ai2': 0, 'draw': 0, 'n': 0}
    while n <= n_games:
        print(f"Game {n} of {n_games}")
        reward = 0
        game_env = GameBoard(3)
        player = 1
        ai1.new_game()
        ai2.new_game()
        while True:
            if player == 1:
                # Sense
                ai1.sense_state(game_env.board)
                # Learn
                if game_env.check_win() == 1:
                    # print(game_env.board)
                    print('Ai_1 Wins')
                    score['ai1'] += 1
                    score['n'] += 1
                    reward = -1
                    ai1.set_reward_learn(reward)
                    break
                elif game_env.check_win() == -1:
                    # print(game_env.board)
                    print('Ai_2 Wins')
                    score['ai2'] += 1
                    score['n'] += 1
                    reward = 1
                    ai1.set_reward_learn(reward)
                    break
                else:
                    if not np.isin(0, game_env.board):
                        # print(game_env.board)
                        print('Its a Draw!')
                        score['draw'] += 1
                        score['n'] += 1
                        ai1.set_reward_learn(reward)
                        break
                    ai1.set_reward_learn(reward)
                # Act
                while True:
                    play = ai1.make_move()
                    if game_env.board[(play - 1) // 3, (play - 1) % 3] == 0:
                        break
                    else:
                        reward = -2
                        ai1.sense_state(np.array([-2]))
                        ai1.set_reward_learn(reward)
                        ai1.sense_state(game_env.board)
                game_env.board[(play - 1) // 3, (play - 1) % 3] = player
                player *= -1
            else:
                # Sense
                ai2.sense_state(game_env.board)
                # Learn
                if game_env.check_win() == 1:
                    # print(game_env.board)
                    print('Ai_1 Wins')
                    score['ai1'] += 1
                    score['n'] += 1
                    reward = -1
                    ai2.set_reward_learn(reward)
                    break
                elif game_env.check_win() == -1:
                    # print(game_env.board)
                    print('Ai_2 Wins')
                    score['ai2'] += 1
                    score['n'] += 1
                    reward = 1
                    ai2.set_reward_learn(reward)
                    break
                else:
                    if not np.isin(0, game_env.board):
                        # print(game_env.board)
                        print('Its a Draw!')
                        score['draw'] += 1
                        score['n'] += 1
                        ai2.set_reward_learn(reward)
                        break
                    ai2.set_reward_learn(reward)
                # Act
                while True:
                    play = ai2.make_move()
                    if game_env.board[(play - 1) // 3, (play - 1) % 3] == 0:
                        break
                    else:
                        reward = -2
                        ai2.sense_state(np.array([-2]))
                        ai2.set_reward_learn(reward)
                        ai2.sense_state(game_env.board)
                game_env.board[(play - 1) // 3, (play - 1) % 3] = player
                player *= -1
        n += 1
        if d == True:
            ai1.brain.decay_exploration()
            ai2.brain.decay_exploration()
    print(score)


env = GameBoard(3)
ai1 = Agent(1)
ai2 = Agent(-1)
# lets_ai_and_ai(100000, ai1, ai2)
# lets_1_and_ai(ai1)
#
# print(len(ai1.brain.qtable))
# print(ai2.brain.qtable)
