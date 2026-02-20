"""
Dynamic Economic Emissions Dispatch (DEED) Environment for Gymnasium.

This module implements the DEED problem as a Gymnasium reinforcement learning
environment. The DEED problem is a multi-objective optimization problem in
power systems where the goal is to schedule 10 generating units over a 24-hour
horizon to minimize both fuel cost and pollutant emissions while satisfying
power demand and operational constraints (ramp rates, generation limits).

The generator data (10 units, cost/emission coefficients, B-matrix loss
coefficients, and 24-hour demand profile) is from a standard IEEE benchmark
test system commonly used in DEED research.

Key fixes over the original implementation:
  - Uses Gymnasium instead of deprecated OpenAI Gym
  - Fixes termination check: tests n==N+1 and m==M BEFORE resetting n
  - Fixes B-matrix loss formula: uses P_j (not P_n) in inner sum
  - Fixes mutable class variables: states_array and p_n_m_df in __init__
  - Fixes reward zeroing bug (self.reward = 0 that discarded computed rewards)
  - Returns proper Gymnasium 5-tuple from step() and 2-tuple from reset()
  - Adds rich Box observation space
  - Adds dense rewards (at each generator assignment, not just end of hour)
  - Removes dead/commented-out code
"""

import math
import sys
from math import sin, exp
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class DEEDEnv(gym.Env):
    """
    Gymnasium environment for the Dynamic Economic Emissions Dispatch problem.

    The agent controls generators 2-10 (9 generators). Generator 1 is the
    "slack" generator whose output is determined by the power balance equation.

    At each step, the agent sets the power output of one generator for the
    current hour by choosing an action in [0, 100], which maps linearly to
    [P_min, P_max] for that generator. After all 9 generators are set for an
    hour, the slack generator is computed, rewards are calculated, and the
    environment advances to the next hour.

    An episode consists of 24 hours x 9 generators = 216 steps.

    Observation (14-dimensional vector):
        [0] current_hour (normalized 0-1)
        [1] current_generator (normalized 0-1)
        [2] demand_this_hour (normalized)
        [3] demand_change (normalized)
        [4] current_gen_prev_power (normalized)
        [5] current_gen_p_min (normalized)
        [6] current_gen_p_max (normalized)
        [7] current_gen_ramp_up (normalized)
        [8] current_gen_ramp_down (normalized)
        [9] total_power_assigned_this_hour (normalized)
        [10] remaining_demand (normalized)
        [11] cumulative_cost (normalized)
        [12] cumulative_emissions (normalized)
        [13] hour_progress (fraction of generators assigned this hour)

    Action: Discrete(101) -- maps to power in [P_min, P_max] for current gen
    """

    metadata = {"render_modes": ["human"]}

    # === IEEE Benchmark Constants ===
    M = 24   # Number of hours in the scheduling horizon
    N = 10   # Number of generating units (1 slack + 9 agent-controlled)

    # Emissions scaling factor (standard in literature)
    E = 10

    # Generator characteristics: 10 units
    # Columns: p_min, p_max, a, b, c, d, e, alpha, beta, gamma, eta, delta, ur, dr
    GEN_DATA = np.array([
        [150, 470, 786.7988, 38.5397, 0.1524, 450, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80],
        [135, 470, 451.3251, 46.1591, 0.1058, 600, 0.036, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80],
        [73,  340, 1049.9977, 40.3965, 0.0280, 320, 0.028, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80, 80],
        [60,  300, 1243.5311, 38.3055, 0.0354, 260, 0.052, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50, 50],
        [73,  243, 1658.5696, 36.3278, 0.0211, 280, 0.063, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50],
        [57,  160, 1356.6592, 38.2704, 0.0179, 310, 0.048, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50],
        [20,  130, 1450.7045, 36.5104, 0.0121, 300, 0.086, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30],
        [47,  120, 1450.7045, 36.5104, 0.0121, 340, 0.082, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30],
        [20,  80,  1455.6056, 39.5804, 0.1090, 270, 0.098, 350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30, 30],
        [10,  55,  1469.4026, 40.5407, 0.1295, 380, 0.094, 360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30, 30],
    ])

    # Column indices for GEN_DATA
    P_MIN, P_MAX = 0, 1
    A, B_COEFF, C, D, E_COEFF = 2, 3, 4, 5, 6
    ALPHA, BETA, GAMMA, ETA, DELTA = 7, 8, 9, 10, 11
    UR, DR = 12, 13

    # B-matrix of transmission line loss coefficients (10x10, symmetric)
    B_MATRIX = np.array([
        [0.000049, 0.000014, 0.000015, 0.000015, 0.000016, 0.000017, 0.000017, 0.000018, 0.000019, 0.000020],
        [0.000014, 0.000045, 0.000016, 0.000016, 0.000017, 0.000015, 0.000015, 0.000016, 0.000018, 0.000018],
        [0.000015, 0.000016, 0.000039, 0.000010, 0.000012, 0.000012, 0.000014, 0.000014, 0.000016, 0.000016],
        [0.000015, 0.000016, 0.000010, 0.000040, 0.000014, 0.000010, 0.000011, 0.000012, 0.000014, 0.000015],
        [0.000016, 0.000017, 0.000012, 0.000014, 0.000035, 0.000011, 0.000013, 0.000013, 0.000015, 0.000016],
        [0.000017, 0.000015, 0.000012, 0.000010, 0.000011, 0.000036, 0.000012, 0.000012, 0.000014, 0.000015],
        [0.000017, 0.000015, 0.000014, 0.000011, 0.000013, 0.000012, 0.000038, 0.000016, 0.000016, 0.000018],
        [0.000018, 0.000016, 0.000014, 0.000012, 0.000013, 0.000012, 0.000016, 0.000040, 0.000015, 0.000016],
        [0.000019, 0.000018, 0.000016, 0.000014, 0.000015, 0.000014, 0.000016, 0.000015, 0.000042, 0.000019],
        [0.000020, 0.000018, 0.000016, 0.000015, 0.000016, 0.000015, 0.000018, 0.000016, 0.000019, 0.000044],
    ])

    # 24-hour power demand profile (MW)
    DEMAND = np.array([
        1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776,
        1924, 2022, 2106, 2150, 2072, 1924, 1776, 1554,
        1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184,
    ], dtype=np.float64)

    # Normalization constants for observations
    MAX_DEMAND = 2200.0
    MAX_POWER = 500.0
    MAX_COST = 100000.0
    MAX_EMISSIONS = 100000.0

    def __init__(
        self,
        Wc: float = 0.5,
        We: float = 0.5,
        Wp: float = 1.0,
        C_penalty: float = 1e6,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the DEED environment.

        Args:
            Wc: Weight for fuel cost in the objective (0 to 1).
            We: Weight for emissions in the objective (0 to 1).
            Wp: Weight for constraint violation penalty.
            C_penalty: Penalty multiplier for constraint violations.
            render_mode: Optional render mode ("human" or None).
        """
        super().__init__()

        self.Wc = Wc
        self.We = We
        self.Wp = Wp
        self.C_penalty = C_penalty
        self.render_mode = render_mode

        # Action space: discrete action 0..100 mapping to power range
        self.action_space = spaces.Discrete(101)

        # Observation space: 14-dimensional normalized vector
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )

        # Instance-level mutable arrays (NOT class-level!)
        # p_schedule[n, m] = power output of generator n at hour m
        self.p_schedule = np.zeros((self.N, self.M), dtype=np.float64)

        # State tracking
        self.m = 0          # Current hour (0-indexed, 0..23)
        self.n = 1          # Current generator (0-indexed, 0..9; agent controls 1..9)
        self.done = False
        self.cumulative_cost = 0.0
        self.cumulative_emissions = 0.0
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        gen_idx = self.n  # 0-indexed generator
        gen = self.GEN_DATA[gen_idx]

        # Previous hour power for this generator
        if self.m == 0:
            prev_power = self.p_schedule[gen_idx, 0]  # initial power
        else:
            prev_power = self.p_schedule[gen_idx, self.m - 1]

        # Total power already assigned this hour (for generators assigned so far)
        total_assigned = 0.0
        for i in range(1, gen_idx):  # generators already assigned this hour
            total_assigned += self.p_schedule[i, self.m]
        # Add slack generator estimate (from previous hour or 0)
        if self.m > 0:
            total_assigned += self.p_schedule[0, self.m - 1]
        else:
            total_assigned += self.p_schedule[0, 0]

        demand = self.DEMAND[self.m]
        demand_change = 0.0 if self.m == 0 else self.DEMAND[self.m] - self.DEMAND[self.m - 1]

        obs = np.array([
            self.m / (self.M - 1),                          # [0] hour (0-1)
            (gen_idx - 1) / max(self.N - 2, 1),             # [1] generator (0-1)
            demand / self.MAX_DEMAND,                        # [2] demand
            (demand_change + 300) / 600.0,                   # [3] demand change (norm)
            prev_power / self.MAX_POWER,                     # [4] prev power
            gen[self.P_MIN] / self.MAX_POWER,                # [5] p_min
            gen[self.P_MAX] / self.MAX_POWER,                # [6] p_max
            gen[self.UR] / 100.0,                            # [7] ramp up
            gen[self.DR] / 100.0,                            # [8] ramp down
            total_assigned / self.MAX_DEMAND,                # [9] total assigned
            (demand - total_assigned) / self.MAX_DEMAND,     # [10] remaining demand
            min(self.cumulative_cost / self.MAX_COST, 1.0),  # [11] cum cost
            min(self.cumulative_emissions / self.MAX_EMISSIONS, 1.0),  # [12] cum emissions
            (gen_idx - 1) / (self.N - 1),                    # [13] hour progress
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _action_to_power(self, gen_idx: int, action: int) -> float:
        """Convert discrete action to power output for a generator."""
        p_min = self.GEN_DATA[gen_idx, self.P_MIN]
        p_max = self.GEN_DATA[gen_idx, self.P_MAX]
        return p_min + (p_max - p_min) * (action / 100.0)

    def _get_constrained_actions(self, gen_idx: int) -> np.ndarray:
        """Get valid actions respecting ramp rate constraints."""
        p_min = self.GEN_DATA[gen_idx, self.P_MIN]
        p_max = self.GEN_DATA[gen_idx, self.P_MAX]
        ur = self.GEN_DATA[gen_idx, self.UR]
        dr = self.GEN_DATA[gen_idx, self.DR]

        if self.m == 0:
            prev_power = self.p_schedule[gen_idx, 0]
        else:
            prev_power = self.p_schedule[gen_idx, self.m - 1]

        power_per_action = (p_max - p_min) / 100.0

        min_power = max(p_min, prev_power - dr)
        max_power = min(p_max, prev_power + ur)

        min_action = max(0, int(np.floor((min_power - p_min) / power_per_action)))
        max_action = min(100, int(np.ceil((max_power - p_min) / power_per_action)))

        return np.arange(min_action, max_action + 1)

    def _compute_slack_power(self, m: int) -> float:
        """
        Compute slack generator (unit 1) power via power balance equation.

        Solves: P_1 + sum(P_n, n=2..N) - P_demand - P_loss = 0
        where P_loss = sum_i sum_j P_i * B_ij * P_j (Kron's loss formula).

        This reduces to a quadratic in P_1:
            B_11 * P_1^2 + (2*sum(B_1j*P_j, j=2..N) - 1) * P_1
            + (P_demand + sum_n sum_j P_n*B_nj*P_j - sum_n P_n) = 0
        for n,j in {2..N}.
        """
        # Sum for linear term: sum(B[0][j] * P_j, j=1..N-1)  (0-indexed j)
        sum_b = 0.0
        for j in range(1, self.N):
            sum_b += self.B_MATRIX[0, j] * self.p_schedule[j, m]

        # Sum for constant term part 1: sum_n sum_j P_n * B[n][j] * P_j
        # FIX: uses P_j (not P_n) in inner loop
        sum_c1 = 0.0
        for n in range(1, self.N):
            P_n = self.p_schedule[n, m]
            for j in range(1, self.N):
                P_j = self.p_schedule[j, m]
                sum_c1 += P_n * self.B_MATRIX[n, j] * P_j

        # Sum for constant term part 2: sum(P_n, n=1..N-1)
        sum_c2 = 0.0
        for n in range(1, self.N):
            sum_c2 += self.p_schedule[n, m]

        # Quadratic coefficients
        a = self.B_MATRIX[0, 0]
        b = 2.0 * sum_b - 1.0
        c = self.DEMAND[m] + sum_c1 - sum_c2

        # Solve quadratic
        discriminant = b ** 2 - 4.0 * a * c
        if discriminant < 0:
            # Fallback: use simple power balance (ignore losses)
            p1 = self.DEMAND[m] - sum_c2
            return p1

        # Take the smaller root (physically meaningful)
        p1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
        return p1

    def _fuel_cost_unit(self, n: int, m: int) -> float:
        """Compute fuel cost for generator n at hour m (0-indexed)."""
        gen = self.GEN_DATA[n]
        p = self.p_schedule[n, m]
        a_n = gen[self.A]
        b_n = gen[self.B_COEFF]
        c_n = gen[self.C]
        d_n = gen[self.D]
        e_n = gen[self.E_COEFF]
        p_min = gen[self.P_MIN]

        # Quadratic + valve-point loading effect
        cost = a_n + b_n * p + c_n * p ** 2 + abs(d_n * sin(e_n * (p_min - p)))
        return cost

    def _fuel_cost_total(self, m: int) -> float:
        """Total fuel cost for all generators at hour m."""
        return sum(self._fuel_cost_unit(n, m) for n in range(self.N))

    def _emissions_unit(self, n: int, m: int) -> float:
        """Compute emissions for generator n at hour m (0-indexed)."""
        gen = self.GEN_DATA[n]
        p = self.p_schedule[n, m]
        alpha = gen[self.ALPHA]
        beta = gen[self.BETA]
        gamma = gen[self.GAMMA]
        eta = gen[self.ETA]
        delta = gen[self.DELTA]

        emissions = self.E * (alpha + beta * p + gamma * p ** 2 + eta * exp(delta * p))
        return emissions

    def _emissions_total(self, m: int) -> float:
        """Total emissions for all generators at hour m."""
        return sum(self._emissions_unit(n, m) for n in range(self.N))

    def _penalty_slack(self, m: int) -> float:
        """
        Compute penalty for slack generator constraint violations.

        Checks:
        1. Generation limits: P_min <= P_1 <= P_max
        2. Ramp rate limits: -DR <= P_1(m) - P_1(m-1) <= UR
        """
        p1 = self.p_schedule[0, m]
        p1_min = self.GEN_DATA[0, self.P_MIN]
        p1_max = self.GEN_DATA[0, self.P_MAX]
        ur1 = self.GEN_DATA[0, self.UR]
        dr1 = self.GEN_DATA[0, self.DR]

        penalty = 0.0

        # Generation limit violation
        if p1 > p1_max:
            penalty += self.C_penalty * abs(p1 - p1_max)
        elif p1 < p1_min:
            penalty += self.C_penalty * abs(p1_min - p1)

        # Ramp rate violation
        if m > 0:
            p1_prev = self.p_schedule[0, m - 1]
        else:
            p1_prev = p1  # No ramp constraint for first hour

        ramp = p1 - p1_prev
        if ramp > ur1:
            penalty += self.C_penalty * abs(ramp - ur1)
        elif ramp < -dr1:
            penalty += self.C_penalty * abs(ramp + dr1)

        return penalty

    def _compute_hour_reward(self, m: int) -> tuple:
        """
        Compute the reward for a completed hour.

        Returns:
            (reward, cost, emissions, penalty)
        """
        cost = self._fuel_cost_total(m)
        emissions = self._emissions_total(m)
        penalty = self._penalty_slack(m)

        # Weighted scalarized objective (negative because we minimize)
        reward = -(self.Wc * cost + self.We * emissions + self.Wp * penalty)

        return reward, cost, emissions, penalty

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple:
        """
        Reset the environment to start a new episode.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        self.m = 0
        self.n = 1   # First agent-controlled generator (0-indexed)
        self.done = False
        self.step_count = 0
        self.cumulative_cost = 0.0
        self.cumulative_emissions = 0.0

        # Initialize power schedule with random feasible dispatch
        self.p_schedule = np.zeros((self.N, self.M), dtype=np.float64)
        for i in range(self.N):
            p_min = self.GEN_DATA[i, self.P_MIN]
            p_max = self.GEN_DATA[i, self.P_MAX]
            self.p_schedule[i, 0] = p_min + (p_max - p_min) * self.np_random.integers(0, 101) / 100.0

        info = {
            "hour": self.m,
            "generator": self.n,
            "demand": self.DEMAND[self.m],
        }

        return self._get_obs(), info

    def step(self, action: int) -> tuple:
        """
        Execute one step: set power for current generator, advance state.

        Args:
            action: Integer 0-100 mapping to power output.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert not self.done, "Episode is done, call reset()"
        assert self.action_space.contains(int(action)), f"Invalid action: {action}"

        action = int(action)
        gen_idx = self.n  # 0-indexed current generator
        reward = 0.0
        info = {}

        # Check ramp rate constraints
        valid_actions = self._get_constrained_actions(gen_idx)

        if action in valid_actions:
            power = self._action_to_power(gen_idx, action)
            self.p_schedule[gen_idx, self.m] = power
        else:
            # Clamp to nearest valid action
            nearest = valid_actions[np.argmin(np.abs(valid_actions - action))]
            power = self._action_to_power(gen_idx, nearest)
            self.p_schedule[gen_idx, self.m] = power
            # Small penalty for choosing invalid action
            reward -= 100.0

        self.step_count += 1

        # Advance to next generator
        self.n += 1

        # Check if all agent-controlled generators are set for this hour
        terminated = False
        truncated = False

        if self.n >= self.N:
            # All generators 1-9 set; compute slack generator
            p1 = self._compute_slack_power(self.m)
            self.p_schedule[0, self.m] = p1

            # Compute hour reward
            hour_reward, cost, emissions, penalty = self._compute_hour_reward(self.m)
            reward += hour_reward
            self.cumulative_cost += cost
            self.cumulative_emissions += emissions

            info["hour_cost"] = cost
            info["hour_emissions"] = emissions
            info["hour_penalty"] = penalty
            info["slack_power"] = p1

            # FIX: Check termination BEFORE advancing hour
            if self.m >= self.M - 1:
                # Last hour completed -- episode done
                terminated = True
                self.done = True
                info["total_cost"] = self.cumulative_cost
                info["total_emissions"] = self.cumulative_emissions
            else:
                # Advance to next hour
                self.m += 1
                self.n = 1  # Reset to first agent-controlled generator

                # Copy previous hour's power as starting point for new hour
                self.p_schedule[:, self.m] = self.p_schedule[:, self.m - 1]
        else:
            # Dense reward: small signal proportional to how much demand is being met
            total_assigned = sum(self.p_schedule[i, self.m] for i in range(1, self.n))
            demand = self.DEMAND[self.m]
            coverage = min(total_assigned / demand, 1.0) if demand > 0 else 0.0
            reward += coverage * 10.0  # Small positive shaping reward

        info["hour"] = self.m
        info["generator"] = self.n if not terminated else -1
        info["step_count"] = self.step_count
        info["cumulative_cost"] = self.cumulative_cost
        info["cumulative_emissions"] = self.cumulative_emissions

        obs = self._get_obs() if not terminated else np.zeros(14, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def get_dispatch_schedule(self) -> pd.DataFrame:
        """Return the current dispatch schedule as a DataFrame."""
        hours = [f"Hour {m+1}" for m in range(self.M)]
        units = [f"Unit {n+1}" for n in range(self.N)]
        return pd.DataFrame(self.p_schedule, index=units, columns=hours)

    def get_hourly_costs(self) -> np.ndarray:
        """Return hourly total fuel costs."""
        return np.array([self._fuel_cost_total(m) for m in range(self.M)])

    def get_hourly_emissions(self) -> np.ndarray:
        """Return hourly total emissions."""
        return np.array([self._emissions_total(m) for m in range(self.M)])

    def render(self):
        """Print current state."""
        if self.render_mode == "human":
            print(f"Hour: {self.m+1}/{self.M}, Generator: {self.n+1}/{self.N}")
            print(f"Demand: {self.DEMAND[self.m]:.1f} MW")
            total = sum(self.p_schedule[i, self.m] for i in range(self.N))
            print(f"Total generation: {total:.1f} MW")
            print(f"Cumulative cost: {self.cumulative_cost:.2f}")
            print(f"Cumulative emissions: {self.cumulative_emissions:.2f}")


def make_deed_env(Wc=0.5, We=0.5, **kwargs):
    """Factory function to create a DEED environment."""
    return DEEDEnv(Wc=Wc, We=We, **kwargs)
