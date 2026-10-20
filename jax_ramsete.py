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


# The firmware rounds the arguments it hands to DriveBase.drive(), because
# Pybricks truncates float arguments toward zero. These constants mirror the
# unit conversions in pybricks_code/ramsete_fast.py.
_MM_PER_M = 1000.0
_DEG_PER_RAD = 180.0 / jnp.pi


def _round_half_away(x):
    """Round half away from zero, matching the firmware's _round()."""
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


def quantize_drive_command(v_cmd, omega_cmd):
    """Round a Ramsete command the way the deployed controller does.

    ``pybricks_code/ramsete_fast.py`` rounds the commanded linear speed
    (mm/s) and turn rate (deg/s) before calling ``DriveBase.drive()``.
    The model applies the same rounding so the twin sees the resolution the
    robot actually gets instead of a continuous command.
    """
    v_mm = _round_half_away(v_cmd * _MM_PER_M)
    omega_deg = _round_half_away(omega_cmd * _DEG_PER_RAD)
    return v_mm / _MM_PER_M, omega_deg / _DEG_PER_RAD


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
        # The pose is read by indexing, not by unpacking the array.  Unpacking
        # a traced array is a JAX "unstack" primitive, and Immrax's natural
        # inclusion has no rule for it, so the closed-loop tick map - which
        # embeds this controller and is evaluated with a Jacobian inclusion -
        # dies with "unstack not in inclusion_registry".  Indexing compiles to
        # slice+squeeze, which the registry has, and it is the same number:
        # the nominal path is unchanged.  (JAX 0.6, which the measurements
        # were taken on, compiled the unpacking to slices; JAX 0.12 does not.)
        x = current_pose[0]
        y = current_pose[1]
        theta = current_pose[2]
        xr = ref_pose[0]
        yr = ref_pose[1]
        thetar = ref_pose[2]
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
    # The firmware rounds the drive() arguments, so the applied command has
    # the same resolution the robot gets.
    v_cmd, omega_cmd = quantize_drive_command(v_cmd, omega_cmd)
    v_actual = smooth_deadband(v_cmd, backlash_b)
    omega_actual = smooth_deadband(omega_cmd, backlash_b * 2.0)
    new_theta = current_pose[2] + omega_actual * dt
    new_x = current_pose[0] + v_actual * jnp.cos(current_pose[2]) * dt
    new_y = current_pose[1] + v_actual * jnp.sin(current_pose[2]) * dt
    return jnp.array([new_x, new_y, new_theta]), v_cmd, omega_cmd
