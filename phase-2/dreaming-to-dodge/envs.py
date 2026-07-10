"""The environments.

PRIMARY (default): `Catch` — a tiny, fast, fully-image-observation game rendered
to 64x64 RGB frames, implemented in pure numpy.  A ball falls one row per step
from the top; a paddle on the bottom row moves left/stay/right; you get +1 if the
paddle is under the ball when it reaches the bottom, else -1.

Why Catch for teaching IRIS:
  * The observation is a real *image*, so the discrete autoencoder and the
    token Transformer do exactly the job they do in IRIS on Atari — just on a
    scene simple enough that a student can *see* whether the world model "gets
    it" (does the dreamed ball fall straight? does the dreamed paddle follow the
    action?).
  * Dynamics are deterministic given the ball's spawn column, so a good world
    model can become near-pixel-perfect -- the clean analogue of IRIS's
    "pixel-perfect Pong" result.
  * Reward is frequent (one terminal +/-1 per ~grid-height steps) and there is a
    KNOWN-OPTIMAL policy (move toward the ball's column), giving an honest,
    unambiguous baseline to beat: random ~= 25-50% catch, oracle = 100%.

FALLBACK / STRETCH: `AtariEnv` wraps ALE/Pong-v5 (grayscale, 64x64) behind the
same interface, wired via configs/pong.yaml.  It is NOT on the critical path;
see CAPSTONE.md for the switch rule.

Every env exposes the same tiny interface:
    reset(seed) -> obs (H,W,3) float32 in [0,1]
    step(action) -> obs, reward (float), done (bool)
    num_actions : int
    render()   -> obs   (current frame)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Headless display bootstrap (for VizDoom's OpenGL renderer).
# On a GPU box with no X server, VizDoom's renderer has nowhere to draw and the
# screen buffer comes back a flat uniform gray.  We spin up a tiny in-container
# X virtual framebuffer (Xvfb) once per process and point DISPLAY at it; combined
# with LIBGL_ALWAYS_SOFTWARE=1 (Mesa llvmpipe) this makes VizDoom render real
# pixels without a GPU-attached display.  No-op if a DISPLAY already exists.
# ---------------------------------------------------------------------------
_XVFB_STARTED = False


def _ensure_display():
    global _XVFB_STARTED
    import os
    if os.environ.get("DISPLAY") or _XVFB_STARTED:
        return
    import subprocess, time
    disp_num = 99
    subprocess.Popen(
        ["Xvfb", f":{disp_num}", "-screen", "0", "640x480x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ["DISPLAY"] = f":{disp_num}"
    # wait (up to ~15s) for the X socket to appear before any DoomGame.init()
    sock = f"/tmp/.X11-unix/X{disp_num}"
    for _ in range(150):
        if os.path.exists(sock):
            break
        time.sleep(0.1)
    time.sleep(0.5)
    _XVFB_STARTED = True


# ---------------------------------------------------------------------------
# Catch  (primary env)
# ---------------------------------------------------------------------------
class Catch:
    """Deterministic pixel Catch on a `grid` x `grid` logical board, rendered
    to a `size` x `size` x 3 RGB frame."""

    num_actions = 3   # 0=left, 1=stay, 2=right

    def __init__(self, grid: int = 8, size: int = 64, seed: int = 0):
        self.grid = grid
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.cell = size // grid
        self.reset(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ball_row = 0
        self.ball_col = int(self.rng.integers(0, self.grid))
        self.paddle = self.grid // 2
        self.done = False
        return self.render()

    def step(self, action: int):
        assert not self.done, "step() called on a finished episode; reset() first"
        # move paddle
        self.paddle = int(np.clip(self.paddle + (action - 1), 0, self.grid - 1))
        # ball falls one row
        self.ball_row += 1
        reward, self.done = 0.0, False
        if self.ball_row >= self.grid - 1:
            self.done = True
            reward = 1.0 if self.paddle == self.ball_col else -1.0
        return self.render(), reward, self.done

    def render(self):
        """Draw ball (white) and paddle (also white, 1 cell wide) on black."""
        img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        c = self.cell

        def draw(r, col, colour):
            y0, x0 = r * c, col * c
            img[y0:y0 + c, x0:x0 + c] = colour

        draw(self.ball_row, self.ball_col, (1.0, 1.0, 1.0))         # ball
        draw(self.grid - 1, self.paddle, (0.4, 0.7, 1.0))          # paddle (blue-ish)
        return img

    # -- oracle & random baselines (for the honest "beat-a-baseline" bar) ------
    def oracle_action(self):
        """Move the paddle toward the ball's column (the optimal policy)."""
        if self.paddle < self.ball_col:
            return 2
        if self.paddle > self.ball_col:
            return 0
        return 1


# ---------------------------------------------------------------------------
# Atari wrapper (stretch / fallback — same interface)
# ---------------------------------------------------------------------------
class AtariEnv:
    """ALE/Pong-v5 (or any ALE game) resized to size x size grayscale->RGB.

    Kept off the critical path; used only if the smoke test shows ale-py builds
    fast and stepping is cheap, and there is budget headroom (see CAPSTONE.md).
    """

    def __init__(self, game: str = "ALE/Pong-v5", size: int = 64, seed: int = 0):
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        self.size = size
        self.env = gym.make(game, frameskip=4, repeat_action_probability=0.0)
        self.num_actions = self.env.action_space.n
        self._seed = seed
        self.reset(seed)

    def _prep(self, frame):
        # frame is (210,160,3) uint8; grayscale + resize to size x size, RGB-stack
        import numpy as np
        g = frame.mean(axis=2)
        # nearest-neighbour resize (no cv2 dependency)
        ys = (np.linspace(0, g.shape[0] - 1, self.size)).astype(int)
        xs = (np.linspace(0, g.shape[1] - 1, self.size)).astype(int)
        small = g[np.ix_(ys, xs)] / 255.0
        return np.repeat(small[:, :, None], 3, axis=2).astype(np.float32)

    def reset(self, seed: int | None = None):
        obs, _ = self.env.reset(seed=seed if seed is not None else self._seed)
        self._last = self._prep(obs)
        return self._last

    def step(self, action: int):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self._last = self._prep(obs)
        return self._last, float(reward), bool(terminated or truncated)

    def render(self):
        return self._last


# ---------------------------------------------------------------------------
# SpatialSIR  (the "Dreaming to Contain" practical env)
# ---------------------------------------------------------------------------
class SpatialSIR:
    """A stochastic spatial-SIR outbreak on a `grid` x `grid` population, rendered
    to a `size` x `size` x 3 RGB frame, with a movable vaccination responder.

    Same tiny interface as `Catch` (reset/step/render/num_actions) so the IRIS
    tokenizer / world model / actor-critic train on it UNCHANGED.  This is the
    real, practical problem the world model is pointed at: the policy learns --
    purely inside *imagined* outbreaks -- WHERE and WHEN to vaccinate to contain a
    disease it has only seen spread a handful of times (you cannot A/B-test a real
    epidemic; that is exactly why a learned world model + planning-in-imagination
    is the right tool).

    State: each of grid*grid individuals is Susceptible / Infected / Recovered /
    Vaccinated.  Dynamics (per step): every infected individual infects each
    susceptible von-Neumann neighbour with prob `beta`, and recovers with prob
    `gamma`; vaccinated & recovered are immune.  The agent moves a responder
    cursor and can vaccinate a patch of susceptibles (S->V), building immune
    barriers -- the analogue of Catch's paddle, but the goal is CONTAINMENT.

    Actions (5): 0=up 1=down 2=left 3=right (move cursor by `cursor_step`),
    4=vaccinate (convert the S cells in a `patch` x `patch` block at the cursor
    to V).  The agent acts FIRST each step, so a timely vaccination protects
    susceptibles before the disease spreads that step -- but only one patch per
    step, so containment requires *foresight* about where the front will go.

    Reward each step = -(new infections)/N, so the return equals minus the final
    infected fraction: maximising it MAXIMISES the population saved.

    Baseline `oracle_action` = reactive RING VACCINATION (vaccinate the
    susceptible ring at the infection front) -- the real epidemiological strategy
    that eradicated smallpox, and the honest bar the imagination policy must meet.

    Note for the tokenizer: unlike Catch (a bright sprite on a black void), here
    the whole frame is coloured cells and the *rare* class is the red infection
    front (a few cells among many green).  So the "don't drop the small thing"
    failure re-appears as a class-imbalance, not a background-dominance -- the
    same fg-weighted-reconstruction lesson, an instructively different cause.
    """

    num_actions = 5
    S, I, R, V = 0, 1, 2, 3
    # distinct hues so the discrete tokenizer must keep the (rare) red front
    COLORS = {
        S: (0.15, 0.65, 0.25),   # susceptible - green
        I: (0.95, 0.10, 0.10),   # infected    - red
        R: (0.30, 0.30, 0.30),   # recovered   - grey
        V: (0.20, 0.45, 0.95),   # vaccinated  - blue
    }
    CURSOR = (1.0, 0.95, 0.20)   # responder outline - yellow

    def __init__(self, grid: int = 16, size: int = 64, beta: float = 0.3,
                 gamma: float = 0.2, init_infected: int = 2, t_max: int = 32,
                 cursor_step: int = 2, patch: int = 3, seed: int = 0):
        self.grid = grid
        self.size = size
        self.beta = beta
        self.gamma = gamma
        self.init_infected = init_infected
        self.t_max = t_max
        self.cursor_step = cursor_step
        self.patch = patch
        self.cell = size // grid
        self.N = grid * grid
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = np.zeros((self.grid, self.grid), dtype=np.int8)   # all S
        # seed a small central cluster of infections (room to spread every way)
        lo, hi = self.grid // 4, 3 * self.grid // 4
        for _ in range(self.init_infected):
            r = int(self.rng.integers(lo, hi))
            c = int(self.rng.integers(lo, hi))
            self.state[r, c] = self.I
        self.cr = self.grid // 2   # responder cursor (logical row, col)
        self.cc = self.grid // 2
        self.t = 0
        self.done = False
        return self.render()

    def _infected_neighbours(self):
        """von-Neumann count of infected neighbours for every cell."""
        inf = (self.state == self.I).astype(np.int16)
        n = np.zeros_like(inf)
        n[1:, :] += inf[:-1, :]     # neighbour above
        n[:-1, :] += inf[1:, :]     # below
        n[:, 1:] += inf[:, :-1]     # left
        n[:, :-1] += inf[:, 1:]     # right
        return n

    def step(self, action: int):
        assert not self.done, "step() called on a finished episode; reset() first"
        action = int(action)
        # 1) agent acts FIRST (a vaccination protects S before spread this step)
        if action == 4:
            h = self.patch // 2
            r0, r1 = max(0, self.cr - h), min(self.grid, self.cr + h + 1)
            c0, c1 = max(0, self.cc - h), min(self.grid, self.cc + h + 1)
            block = self.state[r0:r1, c0:c1]
            block[block == self.S] = self.V           # only susceptibles vaccinate
        else:
            d = self.cursor_step
            if action == 0:   self.cr = max(0, self.cr - d)
            elif action == 1: self.cr = min(self.grid - 1, self.cr + d)
            elif action == 2: self.cc = max(0, self.cc - d)
            elif action == 3: self.cc = min(self.grid - 1, self.cc + d)

        # 2) stochastic SIR dynamics
        k = self._infected_neighbours()
        p_inf = 1.0 - (1.0 - self.beta) ** k          # >=1 infected nbr -> chance
        S_mask = self.state == self.S
        newly_inf = S_mask & (self.rng.random((self.grid, self.grid)) < p_inf)
        I_mask = self.state == self.I
        newly_rec = I_mask & (self.rng.random((self.grid, self.grid)) < self.gamma)
        self.state[newly_inf] = self.I
        self.state[newly_rec] = self.R

        new_infections = int(newly_inf.sum())
        reward = -new_infections / self.N
        self.t += 1
        n_infected = int((self.state == self.I).sum())
        self.done = (n_infected == 0) or (self.t >= self.t_max)
        return self.render(), reward, self.done

    def render(self):
        img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        c = self.cell
        for state_val, colour in self.COLORS.items():
            mask = self.state == state_val
            if mask.any():
                big = np.repeat(np.repeat(mask, c, axis=0), c, axis=1)
                img[big] = colour
        # responder cursor = hollow yellow box over the vaccination patch
        h = self.patch // 2
        r0, c0 = max(0, (self.cr - h) * c), max(0, (self.cc - h) * c)
        r1 = min(self.size, (self.cr + h + 1) * c)
        c1 = min(self.size, (self.cc + h + 1) * c)
        img[r0:r1, c0:c0 + 1] = self.CURSOR
        img[r0:r1, max(0, c1 - 1):c1] = self.CURSOR
        img[r0:r0 + 1, c0:c1] = self.CURSOR
        img[max(0, r1 - 1):r1, c0:c1] = self.CURSOR
        return img

    # -- oracle & metric (the honest "beat-a-baseline" bar) --------------------
    def oracle_action(self):
        """Reactive RING VACCINATION: vaccinate the susceptible cells at the
        infection front; if none are under the cursor, step toward the nearest."""
        if not (self.state == self.I).any():
            return 1
        k = self._infected_neighbours()
        frontier = (self.state == self.S) & (k > 0)   # S touching the front
        if not frontier.any():
            return 1
        fr, fc = np.where(frontier)
        dist = np.abs(fr - self.cr) + np.abs(fc - self.cc)
        j = int(np.argmin(dist))
        dr, dc = int(fr[j]) - self.cr, int(fc[j]) - self.cc
        h = self.patch // 2
        if abs(dr) <= h and abs(dc) <= h:             # patch already covers it
            return 4
        if abs(dr) >= abs(dc):                        # move on the larger axis
            return 1 if dr > 0 else 0
        return 3 if dc > 0 else 2

    ring_action = oracle_action                       # explicit alias

    def saved_fraction(self):
        """Fraction of the population never infected (S + V) — the objective."""
        return float(((self.state == self.S) | (self.state == self.V)).mean())


# ---------------------------------------------------------------------------
# DoomTakeCover  (the "Dreaming to Dodge" practical env)
# ---------------------------------------------------------------------------
class DoomTakeCover:
    """VizDoom's `take_cover` scenario behind the same tiny interface as `Catch`.

    A first-person view of a dungeon: monsters line the far wall and hurl
    fireballs; the agent STRAFES left/right to dodge them and survives as long
    as it can.  This is the exact scenario Ha & Schmidhuber's "World Models"
    (2018) made famous by training a policy *entirely inside its own dream*; we
    reproduce it from scratch with the stronger IRIS recipe (VQ-VAE tokenizer +
    Transformer world model + actor-critic in imagination).

    Observation: the 320x240 game screen downsampled to `size` x `size` x 3 RGB
    in [0,1] (IRIS's native 64x64).  Actions (3): 0=no-op, 1=strafe-left,
    2=strafe-right, each repeated for `frame_skip` game tics (so one agent step
    == `frame_skip` tics; survival in tics == steps * frame_skip).

    Reward = +1 for every step the agent survives, 0 on the step it dies.  So
    the undiscounted return equals the number of steps survived, and maximising
    it maximises survival time -- and under the world model's 3-class reward head
    (sign(r) in {-1,0,+1}) that is class 2 while alive, class 1 on death, making
    the *termination* prediction the signal that matters (unlike Catch's single
    terminal +/-1).  `t_max` caps an episode so evaluation is bounded.

    Why this env for teaching IRIS: the rare-but-lethal detail is the bright
    fireball -- a few bright pixels among a busy scene.  If the tokenizer drops
    it (exactly Catch's dropped-ball / the epidemic's dropped-front failure), the
    dream has no threat and the agent cannot learn to dodge.  Same fg-weighted /
    EMA-codebook lesson, higher stakes.  `oracle_action` is a heuristic vision
    dodger (move away from the brightest incoming blob) -- an honest baseline and
    a deterministic action source for the real-vs-dream fidelity check.

    NB: `vizdoom` is imported lazily inside __init__ so envs.py stays importable
    (for Catch / epidemic) in environments that do not have it installed.
    """

    num_actions = 3   # 0=no-op, 1=strafe-left, 2=strafe-right

    def __init__(self, size: int = 64, frame_skip: int = 4, t_max: int = 525,
                 living_reward: float = 1.0, seed: int = 0):
        import os
        _ensure_display()                 # start Xvfb so the GL renderer has a display
        import vizdoom as vzd
        self.vzd = vzd
        self.size = size
        self.frame_skip = frame_skip
        self.t_max = t_max
        self.living_reward = living_reward

        game = vzd.DoomGame()
        game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
        # force the 64x64-friendly small colour screen, no HUD/weapon clutter so
        # the tokenizer sees a clean scene dominated by walls + fireballs.
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_window_visible(False)
        game.set_render_hud(False)
        game.set_render_weapon(False)
        game.set_render_crosshair(False)
        game.set_render_decals(False)
        game.set_render_messages(False)
        game.set_render_particles(True)   # keep the fireballs visually rich
        game.set_mode(vzd.Mode.PLAYER)
        # pin the action space to strafing so our action map is stable regardless
        # of what the shipped .cfg lists (take_cover uses MOVE_LEFT/MOVE_RIGHT).
        game.clear_available_buttons()
        game.add_available_button(vzd.Button.MOVE_LEFT)
        game.add_available_button(vzd.Button.MOVE_RIGHT)
        game.set_episode_timeout(t_max * frame_skip + frame_skip)  # tics; we also cap in steps
        game.set_seed(int(seed))
        game.init()
        self.game = game
        # button vectors for [MOVE_LEFT, MOVE_RIGHT]
        self._btn = [[0, 0], [1, 0], [0, 1]]   # no-op, left, right
        self.rng = np.random.default_rng(seed)
        self._last = np.zeros((size, size, 3), dtype=np.float32)
        self.done = False
        self.t = 0
        self.reset(seed)

    # -- frame preprocessing: VizDoom screen -> size x size x 3 in [0,1] --------
    def _grab(self):
        state = self.game.get_state()
        if state is None:
            return self._last
        buf = np.asarray(state.screen_buffer)
        if buf.ndim == 3 and buf.shape[0] == 3:        # CHW -> HWC
            buf = np.transpose(buf, (1, 2, 0))
        h, w = buf.shape[:2]
        ys = np.linspace(0, h - 1, self.size).astype(int)
        xs = np.linspace(0, w - 1, self.size).astype(int)
        small = buf[np.ix_(ys, xs)][:, :, :3].astype(np.float32) / 255.0
        self._last = small
        return small

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self.t = 0
        self.done = False
        return self._grab()

    def step(self, action: int):
        assert not self.done, "step() called on a finished episode; reset() first"
        self.game.make_action(self._btn[int(action)], self.frame_skip)
        self.t += 1
        dead = self.game.is_episode_finished()
        timeout = self.t >= self.t_max
        self.done = bool(dead or timeout)
        if not dead:
            self._grab()                       # refresh the live frame
            reward = self.living_reward        # survived this step
        else:
            reward = 0.0                       # died on this step
        return self._last, float(reward), self.done

    def render(self):
        return self._last

    # -- heuristic baseline: strafe away from the brightest incoming blob -------
    def oracle_action(self):
        f = self._last
        band = f[self.size // 3: 7 * self.size // 8, :, :]      # rows ahead of us
        bright = band.max(axis=2)                               # luminance
        warm = np.clip(band[:, :, 0] - 0.4 * band[:, :, 2], 0, 1)  # red/orange emphasis
        threat = (bright * warm).sum(axis=0)                    # per-column (W,)
        if float(threat.max()) < 1e-3:
            return 0                                            # nothing bright -> hold
        center = self.size // 2
        col = int(np.argmax(threat))
        if col < center - 2:
            return 2                                            # threat left -> go right
        if col > center + 2:
            return 1                                            # threat right -> go left
        # threat dead-centre: flee toward the emptier half
        return 2 if threat[:center].sum() >= threat[center:].sum() else 1

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass


def make_env(cfg: dict, seed: int = 0):
    """Factory used by the Modal functions. cfg is a plain dict from a YAML."""
    name = cfg.get("env", "catch")
    if name == "catch":
        return Catch(grid=cfg.get("grid", 8), size=cfg.get("resolution", 64), seed=seed)
    if name in ("epidemic", "sir"):
        return SpatialSIR(grid=cfg.get("grid", 16), size=cfg.get("resolution", 64),
                          beta=cfg.get("beta", 0.3), gamma=cfg.get("gamma", 0.2),
                          init_infected=cfg.get("init_infected", 2),
                          t_max=cfg.get("t_max", 32),
                          cursor_step=cfg.get("cursor_step", 2),
                          patch=cfg.get("patch", 3), seed=seed)
    if name in ("doom", "takecover", "doom_takecover"):
        return DoomTakeCover(size=cfg.get("resolution", 64),
                             frame_skip=cfg.get("frame_skip", 4),
                             t_max=cfg.get("t_max", 525),
                             living_reward=cfg.get("living_reward", 1.0),
                             seed=seed)
    if name == "atari":
        return AtariEnv(game=cfg.get("game", "ALE/Pong-v5"),
                        size=cfg.get("resolution", 64), seed=seed)
    raise ValueError(f"unknown env {name}")
