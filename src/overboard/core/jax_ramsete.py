import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple

def smooth_deadband(v_cmd, b, epsilon=1e-3):
    return v_cmd - (b / 2.0) * jnp.tanh(v_cmd / epsilon)

class JAXRamseteController:
    def __init__(self, b: float = 2.0, zeta: float = 0.7):
        self.b = b
        self.zeta = zeta

    def calculate(self, current_pose: jnp.ndarray, ref_pose: jnp.ndarray, ref_v: float, ref_omega: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, y, theta = current_pose
        xr, yr, thetar = ref_pose
        ex_global = xr - x
        ey_global = yr - y
        ex = jnp.cos(theta) * ex_global + jnp.sin(theta) * ey_global
        ey = -jnp.sin(theta) * ex_global + jnp.cos(theta) * ey_global
        etheta = thetar - theta
        k1 = 2 * self.zeta * jnp.sqrt(ref_omega**2 + self.b * ref_v**2)
        v_cmd = ref_v * jnp.cos(etheta) + k1 * ex

        # Avoid jnp.abs and jnp.where for immrax compatibility if possible,
        # or use a smoother sinc approximation.
        # sinc(x) approx 1 - x^2/6 for small x
        etheta2 = etheta**2
        sinc_etheta = 1.0 - etheta2 / 6.0 + (etheta2**2) / 120.0

        omega_cmd = ref_omega + self.b * ref_v * sinc_etheta * ey + k1 * etheta
        return v_cmd, omega_cmd

@jit
def ramsete_step_jax(current_pose: jnp.ndarray, ref_pose: jnp.ndarray, ref_v: float, ref_omega: float, dt: float,
                     backlash_b: float = 0.0, b: float = 2.0, zeta: float = 0.7):
    controller = JAXRamseteController(b, zeta)
    v_cmd, omega_cmd = controller.calculate(current_pose, ref_pose, ref_v, ref_omega)
    v_actual = smooth_deadband(v_cmd, backlash_b)
    omega_actual = smooth_deadband(omega_cmd, backlash_b * 2.0)
    new_theta = current_pose[2] + omega_actual * dt
    new_x = current_pose[0] + v_actual * jnp.cos(current_pose[2]) * dt
    new_y = current_pose[1] + v_actual * jnp.sin(current_pose[2]) * dt
    return jnp.array([new_x, new_y, new_theta]), v_cmd, omega_cmd
