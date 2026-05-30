import jax
import jax.numpy as jnp
from jax_ramsete import JAXRamseteController, ramsete_step_jax
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

if __name__ == "__main__":
    test_ramsete_equivalence()
    print("Ramsete equivalence test passed!")
