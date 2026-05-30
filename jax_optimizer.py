import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from jax_robot_model import JAXRobotConfig, JAXDifferentialDriveModel
import jaxopt
from typing import Dict, Any, Tuple, Optional
from functools import partial

# --- COST FUNCTIONS ---

def local_trajectory_cost(params, s, e, b, config, num_samples):
    dt = params[0]
    X = params[1:].reshape((10, 5))
    vl, vr = X[:, 3], X[:, 4]
    x, y, theta = X[:, 0], X[:, 1], X[:, 2]
    v = (vl[:-1] + vr[:-1]) / 2.0
    omega = (vr[:-1] - vl[:-1]) / config.track_width
    dx_target = v * jnp.cos(theta[:-1]) * dt
    dy_target = v * jnp.sin(theta[:-1]) * dt
    dtheta_target = omega * dt
    dyn_err_x = x[1:] - (x[:-1] + dx_target)
    dyn_err_y = y[1:] - (y[:-1] + dy_target)
    dyn_err_theta = (theta[1:] - (theta[:-1] + dtheta_target) + jnp.pi) % (2 * jnp.pi) - jnp.pi
    w_dyn = 500.0
    cost = dt * (10 - 1)
    cost += w_dyn * (jnp.sum(jnp.square(dyn_err_x)) + jnp.sum(jnp.square(dyn_err_y)) + jnp.sum(jnp.square(dyn_err_theta)))
    al, ar = (vl[1:] - vl[:-1]) / dt, (vr[1:] - vr[:-1]) / dt
    model = JAXDifferentialDriveModel(config)
    cons = model.check_constraints(vl[:-1], vr[:-1], al, ar)
    w_con = 100.0
    cost += w_con * (jnp.sum(jnp.square(cons["left_motor_violation"])) + jnp.sum(jnp.square(cons["right_motor_violation"])) + jnp.sum(jnp.square(cons["traction_violation"])))
    cost += b['forward_weight'] * jnp.sum(jnp.square(jnp.minimum(0, vl) + jnp.minimum(0, vr)))
    cost += b['reverse_weight'] * jnp.sum(jnp.square(jnp.maximum(0, vl) + jnp.maximum(0, vr)))
    jerk_l = (al[1:] - al[:-1]) / jnp.maximum(dt, 1e-3)
    jerk_r = (ar[1:] - ar[:-1]) / jnp.maximum(dt, 1e-3)
    cost += b['accuracy_weight'] * (jnp.sum(jnp.square(jerk_l)) + jnp.sum(jnp.square(jerk_r)))
    return cost

def local_trajectory_projection(params, hp):
    s, e, config, num_samples = hp
    dt = params[0]
    X = params[1:].reshape((10, 5))
    dt = jnp.clip(dt, 0.001, 1.0)
    X = X.at[0].set(s).at[-1].set(e)
    v_bound = 0.99 * config.max_linear_speed(apply_headroom=True)
    X = X.at[:, 3:5].set(jnp.clip(X[:, 3:5], -v_bound, v_bound))
    return jnp.concatenate([jnp.array([dt]), X.flatten()])

# --- CANDIDATE GENERATION ---

@partial(jit, static_argnums=(2, 3))
def generate_candidates_jax(start_state, end_state, num_samples, num_stomp, config, key):
    N = 10
    dx, dy = end_state[0] - start_state[0], end_state[1] - start_state[1]
    t = jnp.linspace(0, 1, 10)
    x, y = start_state[0] + t * dx, start_state[1] + t * dy
    dtheta = (end_state[2] - start_state[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    theta_lin = start_state[2] + t * dtheta

    # TEB-style seeds
    seeds_theta = jnp.stack([
        theta_lin,             # Standard
        theta_lin + jnp.pi,    # Reverse
        jnp.full(10, start_state[2]), # Point turn start
        jnp.full(10, end_state[2])   # Point turn end
    ])

    def make_guess(theta, v_val):
        # We also seed with non-zero velocities for TEB-style biases
        X = jnp.stack([x, y, theta, jnp.full(10, v_val), jnp.full(10, v_val)], axis=1)
        return jnp.concatenate([jnp.array([0.1]), X.flatten()])

    # Generate Forward/Reverse/Neutral variants
    base_guesses = jnp.concatenate([
        vmap(lambda th: make_guess(th, 0.1))(seeds_theta),
        vmap(lambda th: make_guess(th, -0.1))(seeds_theta),
        vmap(lambda th: make_guess(th, 0.0))(seeds_theta)
    ])

    def perturb(k, base_guess):
        p_X = base_guess[1:].reshape((10, 5)).at[:, :3].add(random.normal(k, (10, 3)) * jnp.array([0.05, 0.05, 0.1]))
        p_X = p_X.at[0, :3].set(start_state[:3]).at[-1, :3].set(end_state[:3])
        return jnp.concatenate([jnp.array([base_guess[0]]), p_X.flatten()])
    stomp_guesses = vmap(lambda k: perturb(k, base_guesses[0]))(random.split(key, num_stomp))
    all_guesses = jnp.concatenate([base_guesses, stomp_guesses])
    total = all_guesses.shape[0]
    biases = {'forward_weight': jnp.zeros(total).at[0].set(10.0), 'reverse_weight': jnp.zeros(total).at[1].set(10.0), 'accuracy_weight': jnp.zeros(total)}
    def eval_one(g, fw, rw, aw):
        b = {'forward_weight': fw, 'reverse_weight': rw, 'accuracy_weight': aw}
        return local_trajectory_cost(g, start_state, end_state, b, config, 10)
    costs = vmap(eval_one)(all_guesses, biases['forward_weight'], biases['reverse_weight'], biases['accuracy_weight'])
    return all_guesses, costs, biases

# --- BATCHED SOLVER ---

def get_jax_refiner(config: JAXRobotConfig):
    @partial(jit, static_argnums=(2, 5))
    def refine_batch(init_guesses, biases, n_samples, s_state, e_state, max_iter):
        # The first argument to the cost function is 'params'.
        # All subsequent arguments are passed via the optimizer's 'run' method.
        # Signature: local_trajectory_cost(params, s, e, b, config, num_samples)
        pg = jaxopt.ProjectedGradient(fun=local_trajectory_cost, projection=local_trajectory_projection, maxiter=max_iter)

        def solve_single(init_guess, bias):
             # Signature of pg.run: params, hyperparams_proj, *args
             # *args corresponds to s, e, b, config, num_samples in local_trajectory_cost
             # hyperparams_proj corresponds to (s, e, config, num_samples) in local_trajectory_projection
             sol = pg.run(init_guess, (s_state, e_state, config, 10), s_state, e_state, bias, config, 10)
             return sol.params, local_trajectory_cost(sol.params, s_state, e_state, bias, config, 10), sol.state.iter_num

        return vmap(solve_single)(init_guesses, biases)
    return refine_batch
