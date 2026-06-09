"""
Mars Rover — starter code for the Lecture 01 assignment.

Build a small Gymnasium environment, then estimate its value function three ways:
  - DP  (value iteration)  : reads the model env.P
  - MC  (Monte Carlo)      : model-free, only env.reset()/env.step()
  - TD(0)                  : model-free, only env.reset()/env.step()

Fill in every TODO. See practice/assignment-mars-rover.pdf for the full spec.
Run:  pip install gymnasium numpy matplotlib  &&  python starter.py
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

GRID = 5
GOAL = (4, 4)
CRATERS = {(2, 2), (1, 3)}
SLIP = 0.1          # prob of slipping to EACH perpendicular direction (intended = 1 - 2*SLIP)
GAMMA = 0.95

MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}     # up, right, down, left
PERP = {0: [3, 1], 1: [0, 2], 2: [1, 3], 3: [2, 0]}        # perpendicular slips per action

def rc(s):  return divmod(s, GRID)
def idx(r, c): return r * GRID + c


class MarsRoverEnv(gym.Env):
    """5x5 dusty crater field. Goal +10, crater -10, each move -1. Modern Gymnasium API."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(GRID * GRID)
        self.action_space = spaces.Discrete(4)
        self.goal = idx(*GOAL)
        self.craters = {idx(*c) for c in CRATERS}
        self.terminals = self.craters | {self.goal}
        self.P = self._build_model()

    def _move(self, s, a):
        r, c = rc(s); dr, dc = MOVES[a]
        return idx(min(max(r + dr, 0), GRID - 1), min(max(c + dc, 0), GRID - 1))

    def _reward(self, s_next):
        # TODO 1: return +10.0 at the goal, -10.0 in a crater, -1.0 otherwise.
        raise NotImplementedError

    def _build_model(self):
        # P[s][a] = [(prob, s_next, reward, terminated), ...]
        P = {s: {a: [] for a in range(4)} for s in range(GRID * GRID)}
        for s in range(GRID * GRID):
            for a in range(4):
                if s in self.terminals:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                outcomes = {}
                # TODO 2: the intended action gets prob (1 - 2*SLIP); each PERP[a] slip gets prob SLIP.
                #         use self._move(s, act) and accumulate: outcomes[s_next] = outcomes.get(s_next, 0) + prob
                raise NotImplementedError
                P[s][a] = [(p, sn, self._reward(sn), sn in self.terminals) for sn, p in outcomes.items()]
        return P

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = idx(0, 0)
        return self.s, {}

    def step(self, action):
        tr = self.P[self.s][action]
        i = self.np_random.choice(len(tr), p=[t[0] for t in tr])
        _, s_next, reward, terminated = tr[i]
        self.s = s_next
        return s_next, reward, terminated, False, {}


# ---------------- Problem 2: DP (uses the model env.P) ----------------
def value_iteration(env, gamma=GAMMA, theta=1e-6):
    """V(s) <- max_a sum_s' P(s'|s,a)[r + gamma V(s')]  until it stops moving. Return (V, policy)."""
    # TODO: implement value iteration over env.P; then extract the greedy policy[s] = argmax_a q_a.
    raise NotImplementedError


# ---------------- Problem 3: MC and TD (model-free, only env.step()) ----------------
def mc_prediction(env, policy, episodes=20000, gamma=GAMMA, alpha=0.02, max_steps=100):
    """Run episodes under `policy`; for each, compute return G and do V(s) <- V(s) + alpha[G - V(s)]."""
    # TODO: implement constant-alpha Monte Carlo prediction.
    raise NotImplementedError


def td_prediction(env, policy, episodes=20000, gamma=GAMMA, alpha=0.05, max_steps=100):
    """Update every step: V(s) <- V(s) + alpha[r + gamma V(s') - V(s)] (bootstrap term is 0 at a terminal s')."""
    # TODO: implement TD(0) prediction.
    raise NotImplementedError


if __name__ == "__main__":
    env = MarsRoverEnv()
    s0, info = env.reset(seed=0)
    print("reset ->", (s0, info))
    s1, r, term, trunc, _ = env.step(env.action_space.sample())
    print("step  ->", (s1, r, term, trunc))

    V, pi = value_iteration(env)
    print("\nDP value grid:\n", np.round(V.reshape(GRID, GRID), 2))
    arrows = {0: "^", 1: ">", 2: "v", 3: "<", None: "."}
    print("greedy policy:\n", np.array([arrows[pi[s]] for s in range(GRID*GRID)]).reshape(GRID, GRID))

    print("\nMC :\n", np.round(mc_prediction(env, pi).reshape(GRID, GRID), 2))
    print("\nTD :\n", np.round(td_prediction(env, pi).reshape(GRID, GRID), 2))
