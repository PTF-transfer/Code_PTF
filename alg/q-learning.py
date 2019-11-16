import numpy as np

class QL:
    def __init__(self, n_feature, n_action,
                 learning_rate=0.01,
                 reward_decay=0.95,
                 e_greedy=1,
                 e_greedy_increment=1e-05):
        self.n_feature = n_feature
        self.n_action = n_action
        self.Q = np.zeros((n_feature, n_action))
        self.learning_rate = learning_rate
        self.e_greedy = e_greedy
        self.reward_decay = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon_increment = e_greedy_increment

    def choose_action(self, s):
        q = self.Q[s]
        if np.random.uniform() < self.epsilon:
            return np.argmax(q)
        else:
            return np.random.randint(0, self.n_action)

    def update(self, s, a, r, s_):
        self.Q[s, a] = (1 - self.learning_rate) * self.Q[s, a] + self.learning_rate * (
                    r + self.reward_decay * np.max(self.Q[s_]))

    def update_epsilon(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

