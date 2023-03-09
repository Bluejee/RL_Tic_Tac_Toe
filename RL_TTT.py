"""
This contains the classes and functions that are necessary for the RL for the Tic Tac Toe Game.
"""
import numpy as np
import json


class GameBoard:
    def __init__(self, board_size, player_x, player_y):
        self.board_size = board_size
        self.board = np.zeros([board_size, board_size], dtype=int)
        self.state = ''
        self.hash_board()
        self.players = {'X': player_x, 'Y': player_y}
        # 1s Represent Xs and -1s Represent Os

    def hash_board(self):
        self.state = ''.join(self.board.flatten().astype(str))

    def check_state(self):
        """
        This function checks and returns the winner, draw or if the game is still in play.
        it checks of there exists any row or column or either of the diagonals has all Xs(1s) or Os(-1s).

        :return: Winner or draw
        """
        # check rows and columns
        for i in range(self.board_size):
            if (np.sum(self.board[:, i]) == self.board_size) or (np.sum(self.board[i, :]) == self.board_size):
                return 'X'
            if (np.sum(self.board[:, i]) == -self.board_size) or (np.sum(self.board[i, :]) == -self.board_size):
                return 'O'

        # check diagonals.
        if (np.sum(np.diag(self.board)) == self.board_size) or (
                np.sum(np.diag(np.fliplr(self.board))) == self.board_size):
            return 'X'
        if (np.sum(np.diag(self.board)) == -self.board_size) or (
                np.sum(np.diag(np.fliplr(self.board))) == -self.board_size):
            return 'O'

        # Now that we know both X and O has not won, the game can either be in play or a draw.
        if np.isin(0, self.board):
            return 'Play'
        else:
            return 'Draw'

    def is_available(self, position):
        if self.board[(position - 1) // 3, (position - 1) % 3] == 0:
            return True
        else:
            return False

    def set_position(self, player_symbol, position):
        if player_symbol == 'X':
            player_symbol = 1
        else:
            player_symbol = -1

        self.board[(position - 1) // 3, (position - 1) % 3] = player_symbol

        # recalculating the state hash.

    def reset_board(self):
        self.board = np.zeros([self.board_size, self.board_size])

    def play_game(self):
        self.reset_board()
        game_state = self.check_state()
        # Player X always plays first.
        while game_state == 'Play':
            # Both players observe, and act according to the state they perceive.
            for current_player in ['X', 'O']:
                # Every time a player senses a state, They make a move and observes the changed state.
                # They then learns about the previous state and action.
                # Learning about the previous state is important.
                # In the Q-Learning algorithm, the value is updated using 2 factors. the reward and the expected reward
                # from the next state.
                # It is easier to let the game calculate the next state than have the player calculate it themselves.

                # If the game ends, they sense an 'END' and learn about their previous action.
                # Which is same as the current action for X.
                # The only exception is if X wins. Then, O learns about its previous action.

                # This check is primarily for the player playing O
                if self.check_state() == 'X':
                    # i.e. if the game is won by X. (Draw need not be checked as O will always be the one drawing.)
                    # O need not play and the loop moves to giving rewards.
                    # self.players[current_player].sense_state('END')
                    break
                else:
                    # This happens if the game is still in play, i.e. if X or O is to make a move.
                    self.players[current_player].sense_state(self.state)

                    # The current_player takes a decision/Acts based on its knowledge and observation.
                    while True:
                        action = self.players[current_player].get_action()
                        if self.is_available(action):
                            break
                    self.set_position(current_player, action)

                    # The state of the Board/Environment Changes.
                    # The players turn ends and the player switches automatically.
                    # If the player is X then it becomes O
                    # Else the round ends.

                # In case the player was O and the last turn of X had caused the game to end. O Simply senses
                # the state as END.

            # Both players have seen the board and made their move.
            game_state = self.check_state()
            # If X won the game, O has already set its game state as END. it will be learning that its previous
            # move was bad.
            if game_state == 'X':
                self.players['X'].sense_state('END')
                x_reward = 1
                o_reward = -1

            # If O won or Drew the game, we have to set the state of X and O as end so that they can learn about their
            # last move.
            elif game_state == 'O':
                x_reward = -1
                o_reward = 1
            elif game_state == 'Draw':
                x_reward = 0
                o_reward = 0

                # If the game is still in play the players just get 0 reward and the game continues.

                # Once both players are done playing. The game evaluates and gives reward to the players.
                # This eliminates the win check for one player when the other has won.

                self.players[current_player].sense_change(self.state)
                # Gets a reward based on the change.
                self.check_state()

                # The player updates/Learns the decision of the previous action, for the previous state.


        else:
        # We now know that the game is not in play.
        if game_state == 'X':
            print('X Wins!')

        elif game_state == 'O':
            print('O Wins!')

        else:
            print('The game is a Draw!')


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
