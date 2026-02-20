"""
Tests for the DEED (Dynamic Economic Emissions Dispatch) environment.

These tests verify that the Gymnasium environment is correctly implemented,
including observation/action spaces, episode lifecycle, constraint enforcement,
and numerical correctness of cost/emissions functions.
"""

import math
import sys
import numpy as np
import pytest

sys.path.insert(0, "/tmp/research-work/jupyterNotebooks")
from deed_env import DEEDEnv


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def env():
    """Create a fresh DEED environment for each test."""
    e = DEEDEnv(Wc=0.5, We=0.5)
    return e


@pytest.fixture
def env_reset(env):
    """Create and reset a DEED environment."""
    obs, info = env.reset(seed=42)
    return env, obs, info


# ============================================================
# Environment creation and structure
# ============================================================

class TestEnvironmentCreation:
    """Tests for environment initialization."""

    def test_creation(self, env):
        """Environment should be created without errors."""
        assert env is not None

    def test_action_space(self, env):
        """Action space should be Discrete(101)."""
        assert env.action_space.n == 101

    def test_observation_space_shape(self, env):
        """Observation space should be Box(14,)."""
        assert env.observation_space.shape == (14,)

    def test_observation_space_bounds(self, env):
        """Observation space should be bounded [0, 1]."""
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    def test_generator_data_shape(self, env):
        """Generator data should have 10 units x 14 columns."""
        assert env.GEN_DATA.shape == (10, 14)

    def test_b_matrix_shape(self, env):
        """B matrix should be 10x10."""
        assert env.B_MATRIX.shape == (10, 10)

    def test_b_matrix_symmetric(self, env):
        """B matrix should be symmetric."""
        np.testing.assert_array_almost_equal(env.B_MATRIX, env.B_MATRIX.T)

    def test_demand_profile(self, env):
        """Demand profile should have 24 hours."""
        assert len(env.DEMAND) == 24
        assert np.all(env.DEMAND > 0)

    def test_custom_weights(self):
        """Custom weights should be stored correctly."""
        env = DEEDEnv(Wc=0.3, We=0.7)
        assert env.Wc == 0.3
        assert env.We == 0.7


# ============================================================
# Reset
# ============================================================

class TestReset:
    """Tests for environment reset."""

    def test_reset_returns_tuple(self, env):
        """Reset should return (obs, info) tuple."""
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_shape(self, env):
        """Reset observation should match observation space."""
        obs, info = env.reset(seed=42)
        assert obs.shape == (14,)
        assert env.observation_space.contains(obs)

    def test_reset_info_dict(self, env):
        """Reset info should contain required keys."""
        obs, info = env.reset(seed=42)
        assert "hour" in info
        assert "generator" in info
        assert "demand" in info

    def test_reset_initial_state(self, env):
        """After reset, hour should be 0 and generator should be 1."""
        env.reset(seed=42)
        assert env.m == 0
        assert env.n == 1
        assert env.done is False
        assert env.step_count == 0

    def test_reset_with_seed_reproducible(self):
        """Reset with same seed should produce same observation."""
        env1 = DEEDEnv()
        env2 = DEEDEnv()
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_clears_cumulative(self, env):
        """Reset should clear cumulative cost and emissions."""
        env.reset(seed=42)
        # Run some steps
        for _ in range(20):
            env.step(env.action_space.sample())
        # Reset and check
        env.reset(seed=42)
        assert env.cumulative_cost == 0.0
        assert env.cumulative_emissions == 0.0


# ============================================================
# Step
# ============================================================

class TestStep:
    """Tests for environment step function."""

    def test_step_returns_5_tuple(self, env_reset):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env, _, _ = env_reset
        result = env.step(50)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_observation_valid(self, env_reset):
        """Step observation should be in observation space."""
        env, _, _ = env_reset
        obs, _, terminated, _, _ = env.step(50)
        if not terminated:
            assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env_reset):
        """Reward should be a finite number."""
        env, _, _ = env_reset
        _, reward, _, _, _ = env.step(50)
        assert isinstance(reward, (int, float, np.floating))
        assert np.isfinite(reward)

    def test_step_terminated_is_bool(self, env_reset):
        """Terminated flag should be boolean."""
        env, _, _ = env_reset
        _, _, terminated, _, _ = env.step(50)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, env_reset):
        """Truncated flag should be boolean."""
        env, _, _ = env_reset
        _, _, _, truncated, _ = env.step(50)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self, env_reset):
        """Info should be a dictionary."""
        env, _, _ = env_reset
        _, _, _, _, info = env.step(50)
        assert isinstance(info, dict)


# ============================================================
# Full episode
# ============================================================

class TestFullEpisode:
    """Tests for running complete episodes."""

    def test_episode_completes(self, env):
        """A full episode should complete in exactly 216 steps."""
        env.reset(seed=42)
        steps = 0
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if steps > 300:
                pytest.fail("Episode did not terminate within 300 steps")
        assert steps == 216  # 24 hours * 9 generators

    def test_episode_total_cost_positive(self, env):
        """Total cost should be positive."""
        env.reset(seed=42)
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(50)
        assert info.get("total_cost", 0) > 0

    def test_episode_total_emissions_positive(self, env):
        """Total emissions should be positive."""
        env.reset(seed=42)
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(50)
        assert info.get("total_emissions", 0) > 0

    def test_rewards_finite_throughout(self, env):
        """All rewards should be finite (not NaN or inf)."""
        env.reset(seed=42)
        terminated = False
        while not terminated:
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Non-finite reward: {reward}"

    def test_multiple_episodes(self, env):
        """Multiple episodes should run without issues."""
        for episode in range(3):
            env.reset(seed=episode)
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env.step(env.action_space.sample())


# ============================================================
# Cost and emissions functions
# ============================================================

class TestCostFunctions:
    """Tests for cost and emissions calculations."""

    def test_fuel_cost_unit_positive(self, env):
        """Fuel cost for each unit should be positive."""
        env.reset(seed=42)
        # Set some power values
        for n in range(env.N):
            p_min = env.GEN_DATA[n, env.P_MIN]
            p_max = env.GEN_DATA[n, env.P_MAX]
            env.p_schedule[n, 0] = (p_min + p_max) / 2.0
        for n in range(env.N):
            cost = env._fuel_cost_unit(n, 0)
            assert cost > 0, f"Unit {n} cost should be positive, got {cost}"

    def test_fuel_cost_with_known_input(self, env):
        """Test fuel cost computation with known generator data."""
        env.reset(seed=42)
        # Unit 1 (index 0) at mid-range
        p = 310.0  # midpoint of [150, 470]
        env.p_schedule[0, 0] = p
        gen = env.GEN_DATA[0]
        a, b, c, d, e_coeff, p_min = (
            gen[env.A], gen[env.B_COEFF], gen[env.C],
            gen[env.D], gen[env.E_COEFF], gen[env.P_MIN]
        )
        expected = a + b * p + c * p ** 2 + abs(d * math.sin(e_coeff * (p_min - p)))
        actual = env._fuel_cost_unit(0, 0)
        assert abs(actual - expected) < 1e-6

    def test_emissions_unit_positive(self, env):
        """Emissions for each unit should be positive."""
        env.reset(seed=42)
        for n in range(env.N):
            p_min = env.GEN_DATA[n, env.P_MIN]
            p_max = env.GEN_DATA[n, env.P_MAX]
            env.p_schedule[n, 0] = (p_min + p_max) / 2.0
        for n in range(env.N):
            em = env._emissions_unit(n, 0)
            assert em > 0, f"Unit {n} emissions should be positive, got {em}"

    def test_total_cost_equals_sum(self, env):
        """Total cost should equal sum of individual unit costs."""
        env.reset(seed=42)
        for n in range(env.N):
            p_min = env.GEN_DATA[n, env.P_MIN]
            env.p_schedule[n, 0] = p_min + 10.0
        total = env._fuel_cost_total(0)
        individual_sum = sum(env._fuel_cost_unit(n, 0) for n in range(env.N))
        assert abs(total - individual_sum) < 1e-6


# ============================================================
# Power balance and constraints
# ============================================================

class TestConstraints:
    """Tests for constraint enforcement."""

    def test_slack_power_balance(self, env):
        """Slack generator should approximately satisfy demand."""
        env.reset(seed=42)
        # Set generators 1-9 to mid-range
        for n in range(1, env.N):
            p_min = env.GEN_DATA[n, env.P_MIN]
            p_max = env.GEN_DATA[n, env.P_MAX]
            env.p_schedule[n, 0] = (p_min + p_max) / 2.0

        p1 = env._compute_slack_power(0)
        env.p_schedule[0, 0] = p1

        # Check power balance: sum(P_i) - P_demand - P_loss ~= 0
        total_gen = sum(env.p_schedule[n, 0] for n in range(env.N))
        demand = env.DEMAND[0]
        # Allow for transmission losses
        assert total_gen > demand * 0.9, "Total generation too low"
        assert total_gen < demand * 1.5, "Total generation too high"

    def test_ramp_rate_constrained_actions(self, env):
        """Constrained actions should respect ramp rates."""
        env.reset(seed=42)
        gen_idx = 1
        # Set previous power to mid-range
        p_min = env.GEN_DATA[gen_idx, env.P_MIN]
        p_max = env.GEN_DATA[gen_idx, env.P_MAX]
        ur = env.GEN_DATA[gen_idx, env.UR]
        dr = env.GEN_DATA[gen_idx, env.DR]
        prev_power = (p_min + p_max) / 2.0
        env.p_schedule[gen_idx, 0] = prev_power

        # Get constrained actions for hour 1
        env.m = 1
        env.p_schedule[gen_idx, 0] = prev_power
        valid = env._get_constrained_actions(gen_idx)

        # All valid actions should map to power within ramp limits
        for act in valid:
            power = env._action_to_power(gen_idx, act)
            assert power >= prev_power - dr - 1.0, f"Below ramp-down: {power}"
            assert power <= prev_power + ur + 1.0, f"Above ramp-up: {power}"

    def test_generation_limits(self, env):
        """Action 0 should give P_min, action 100 should give P_max."""
        for n in range(env.N):
            p_min = env.GEN_DATA[n, env.P_MIN]
            p_max = env.GEN_DATA[n, env.P_MAX]
            assert abs(env._action_to_power(n, 0) - p_min) < 1e-6
            assert abs(env._action_to_power(n, 100) - p_max) < 1e-6

    def test_penalty_zero_when_feasible(self, env):
        """Penalty should be zero when slack generator is within limits."""
        env.reset(seed=42)
        # Set a feasible dispatch
        env.p_schedule[0, 0] = 300.0  # within [150, 470]
        env.p_schedule[0, 1] = 320.0  # ramp of 20 within UR=80
        env.m = 1
        penalty = env._penalty_slack(1)
        assert penalty == 0.0, f"Expected zero penalty, got {penalty}"


# ============================================================
# Dispatch schedule helpers
# ============================================================

class TestHelpers:
    """Tests for helper methods."""

    def test_get_dispatch_schedule(self, env):
        """Dispatch schedule should be a proper DataFrame."""
        env.reset(seed=42)
        df = env.get_dispatch_schedule()
        assert df.shape == (10, 24)
        assert "Unit 1" in df.index
        assert "Hour 1" in df.columns

    def test_get_hourly_costs(self, env):
        """Hourly costs should be an array of length 24."""
        env.reset(seed=42)
        costs = env.get_hourly_costs()
        assert len(costs) == 24

    def test_get_hourly_emissions(self, env):
        """Hourly emissions should be an array of length 24."""
        env.reset(seed=42)
        emissions = env.get_hourly_emissions()
        assert len(emissions) == 24
