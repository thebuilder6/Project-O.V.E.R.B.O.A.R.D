import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Optional, Tuple

def smooth_deadband(v_cmd, b, epsilon=1e-3):
    return v_cmd - (b / 2.0) * jnp.tanh(v_cmd / epsilon)


# Chebyshev coefficients for sinc(x) on x in [-pi, pi], expressed in
# u = 2*(x/pi)^2 - 1. The maximum absolute error is about 2.5e-9 (float64).
# This is used instead of sin(x)/x or jnp.sinc because Immrax's natural
# inclusion does not implement jnp.where, abs, or sinc. The polynomial is
# even, so the same coefficients also match the exact controller for the
# small heading errors that occur in a stable closed loop.
_SINC_CHEB = (
    4.28930948e-01,
    -4.95478386e-01,
    7.09060458e-02,
    -4.51782748e-03,
    1.62945403e-04,
    -3.78552404e-06,
    6.18166893e-08,
)


def _sinc_poly(x):
    """Polynomial approximation of sin(x)/x, valid for |x| <= pi."""
    u = 2.0 * (x ** 2) / (jnp.pi ** 2) - 1.0
    b1 = jnp.zeros_like(u)
    b2 = jnp.zeros_like(u)
    for c in reversed(_SINC_CHEB[1:]):
        b0 = 2.0 * u * b1 - b2 + c
        b2 = b1
        b1 = b0
    return u * b1 - b2 + _SINC_CHEB[0]


def _normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


class JAXRamseteController:
    def __init__(self, b: float = 2.0, zeta: float = 0.7):
        self.b = b
        self.zeta = zeta

    def calculate(self, current_pose: jnp.ndarray, ref_pose: jnp.ndarray,
                  ref_v: float, ref_omega: float,
                  ref_k: Optional[float] = None,
                  normalize: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Ramsete command for one state/reference pair.

        ``ref_k`` is the precomputed sqrt(omega^2 + b*v^2) that the deployed
        controller interpolates between trajectory samples. Pass it when
        following a trajectory; omit it for a single reference point.

        ``normalize=False`` is used by the interval validator, where headings
        are kept continuous (unrolled). It avoids jnp.mod, which Immrax's
        natural inclusion does not implement.
        """
        x, y, theta = current_pose
        xr, yr, thetar = ref_pose
        ex_global = xr - x
        ey_global = yr - y
        ex = jnp.cos(theta) * ex_global + jnp.sin(theta) * ey_global
        ey = -jnp.sin(theta) * ex_global + jnp.cos(theta) * ey_global

        etheta = thetar - theta
        if normalize:
            etheta = _normalize_angle(etheta)

        if ref_k is None:
            k1 = 2 * self.zeta * jnp.sqrt(ref_omega**2 + self.b * ref_v**2)
        else:
            k1 = 2 * self.zeta * ref_k

        v_cmd = ref_v * jnp.cos(etheta) + k1 * ex
        omega_cmd = (ref_omega + self.b * ref_v * _sinc_poly(etheta) * ey
                     + k1 * etheta)
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
