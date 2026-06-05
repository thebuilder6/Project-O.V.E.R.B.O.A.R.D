import unittest
import jax
import jax.numpy as jnp
from overboard.core.jax_ramsete import JAXRamseteController, ramsete_step_jax
import numpy as np

# We'll use a mock of the original controller to compare
class MockRamsete:
    def __init__(self, b=2.0, zeta=0.7):
        self.b = b
        self.zeta = zeta
    def calculate(self, x, y, theta, xr, yr, thetar, ref_v, ref_omega):
        ex_g = xr - x
        ey_g = yr - y
        ex = np.cos(theta) * ex_g + np.sin(theta) * ey_g
        ey = -np.sin(theta) * ex_g + np.cos(theta) * ey_g
        etheta = (thetar - theta + np.pi) % (2 * np.pi) - np.pi
        k1 = 2 * self.zeta * np.sqrt(ref_omega**2 + self.b * ref_v**2)
        v_cmd = ref_v * np.cos(etheta) + k1 * ex
        sinc = np.sin(etheta)/etheta if abs(etheta) > 1e-6 else 1.0
        om_cmd = ref_omega + self.b * ref_v * sinc * ey + k1 * etheta
        return v_cmd, om_cmd

def test_ramsete_equivalence():
    b, zeta = 2.0, 0.7
    jax_controller = JAXRamseteController(b, zeta)
    orig_controller = MockRamsete(b, zeta)

    current_pose = jnp.array([0.1, 0.1, 0.2])
    ref_pose = jnp.array([0.15, 0.12, 0.25])
    ref_v, ref_omega = 0.5, 0.1

    v_jax, om_jax = jax_controller.calculate(current_pose, ref_pose, ref_v, ref_omega)
    v_orig, om_orig = orig_controller.calculate(0.1, 0.1, 0.2, 0.15, 0.12, 0.25, 0.5, 0.1)

    print(f"JAX: v={v_jax:.6f}, om={om_jax:.6f}")
    print(f"Orig: v={v_orig:.6f}, om={om_orig:.6f}")

    assert np.allclose(float(v_jax), v_orig, atol=1e-6)
    assert np.allclose(float(om_jax), om_orig, atol=1e-6)

# --- NEW: Boundary test for the -pi/pi unrolling behavior ---
def test_ramsete_boundary():
    """Prove that unrolled continuous angles yield equivalent results to wrapped modulo angles."""
    b, zeta = 2.0, 0.7
    jax_controller = JAXRamseteController(b, zeta)
    orig_controller = MockRamsete(b, zeta)
    
    # JAX uses unrolled continuous angles (e.g., 3.2 rad is just past pi)
    current_pose = jnp.array([0.0, 0.0, 3.1])
    ref_pose_jax = jnp.array([0.0, 0.0, 3.2])

    # Mock uses raw wrapped angles (-3.083 is wrapped equivalent of 3.2 rad)
    ref_pose_orig = jnp.array([0.0, 0.0, 3.2 - 2 * np.pi])

    ref_v, ref_omega = 0.5, 0.1

    v_jax, om_jax = jax_controller.calculate(current_pose, ref_pose_jax, ref_v, ref_omega)
    v_orig, om_orig = orig_controller.calculate(
        0.0, 0.0, 3.1,
        0.0, 0.0, ref_pose_orig[2],
        0.5, 0.1
    )

    print(f"Boundary JAX: v={v_jax:.6f}, om={om_jax:.6f}")
    print(f"Boundary Orig: v={v_orig:.6f}, om={om_orig:.6f}")

    assert np.allclose(float(v_jax), v_orig, atol=1e-5)
    assert np.allclose(float(om_jax), om_orig, atol=1e-5)
# ------------------------------------------------------------

if __name__ == "__main__":
    test_ramsete_equivalence()
    test_ramsete_boundary()
    print("Ramsete equivalence test passed!")

class TestJAXRamsete(unittest.TestCase):
    def test_ramsete_equivalence(self):
        test_ramsete_equivalence()

    def test_ramsete_boundary(self):
        test_ramsete_boundary()

if __name__ == "__main__":
    unittest.main()
