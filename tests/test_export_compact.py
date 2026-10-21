"""
Tests for the packed Pybricks trajectory export.

The firmware decoder itself cannot run on the host (it imports pybricks and
umath), but the packed format is plain little-endian data, so the host can
round-trip it with struct and execute the generated Python module.
"""

import json
import math
import os
import struct
import tempfile
import unittest

from overboard.deploy.export import COMPACT_HEADER_SIZE, COMPACT_RECORD_SIZE, COMPACT_VERSION, encode_compact_trajectory, write_python_file

_CONFIG = {
    "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
    "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
    "gearing": 1.0, "cof": 1.37,
}


def _samples():
    return [
        {
            "t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0,
            "vl": 0.0, "vr": 0.0, "omega": 0.0,
        },
        {
            "t": 0.1, "x": 0.05, "y": 0.01, "heading": 0.2,
            "vl": 0.4, "vr": 0.6, "omega": 2.0, "event": "go",
        },
        {
            "t": 0.2, "x": 0.12, "y": 0.03, "heading": math.pi - 0.01,
            "vl": -0.2, "vr": -0.1, "omega": -1.0,
        },
    ]


def _decode(blob):
    version, count = struct.unpack_from("<HH", blob, 0)
    rows = [
        struct.unpack_from("<6f", blob, COMPACT_HEADER_SIZE + i * COMPACT_RECORD_SIZE)
        for i in range(count)
    ]
    return version, count, rows


class TestCompactExport(unittest.TestCase):

    def _write_traj(self, tmp, samples):
        path = os.path.join(tmp, "traj.json")
        with open(path, "w") as f:
            json.dump({
                "trajectory": {
                    "config": _CONFIG,
                    "samples": samples,
                }
            }, f)
        return path

    def _exec_export(self, traj, **kwargs):
        out = os.path.join(os.path.dirname(traj), "traj.py")
        write_python_file(traj, out, **kwargs)
        namespace = {}
        with open(out) as f:
            exec(f.read(), namespace)
        return namespace

    def test_pack_roundtrip(self):
        blob, events = encode_compact_trajectory(_samples())

        self.assertEqual(len(blob), COMPACT_HEADER_SIZE + 3 * COMPACT_RECORD_SIZE)
        version, count, rows = _decode(blob)
        self.assertEqual(version, COMPACT_VERSION)
        self.assertEqual(count, 3)
        self.assertEqual(events, ((1, "go"),))

        for source, row in zip(_samples(), rows):
            expected_v = (source["vl"] + source["vr"]) / 2.0
            self.assertAlmostEqual(row[0], source["t"], places=5)
            self.assertAlmostEqual(row[1], source["x"], places=5)
            self.assertAlmostEqual(row[2], source["y"], places=5)
            self.assertAlmostEqual(row[3], source["heading"], places=5)
            self.assertAlmostEqual(row[4], expected_v, places=5)
            self.assertAlmostEqual(row[5], source["omega"], places=5)

    def test_compact_python_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj = self._write_traj(tmp, _samples())
            namespace = self._exec_export(traj, compact=True)

        self.assertIsInstance(namespace["samples"], bytes)
        self.assertEqual(namespace["events"], ((1, "go"),))
        self.assertAlmostEqual(namespace["config"]["cof"], 1.37)

        version, count, rows = _decode(namespace["samples"])
        self.assertEqual(version, COMPACT_VERSION)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(rows[1][4], 0.5, places=5)

    def test_legacy_python_export_still_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj = self._write_traj(tmp, _samples())
            namespace = self._exec_export(traj)

        self.assertIsInstance(namespace["samples"], list)
        self.assertIsInstance(namespace["samples"][0], dict)
        self.assertEqual(namespace["samples"][1]["event"], "go")

    def test_heading_near_pi_roundtrip(self):
        samples = _samples()
        samples[1]["heading"] = 3.13
        samples[2]["heading"] = -3.13
        blob, _ = encode_compact_trajectory(samples)

        _, _, rows = _decode(blob)
        self.assertAlmostEqual(rows[1][3], 3.13, places=5)
        self.assertAlmostEqual(rows[2][3], -3.13, places=5)

    def test_large_trajectory_size(self):
        samples = [{
            "t": i * 0.02, "x": 0.0, "y": 0.0, "heading": 0.0,
            "vl": 0.0, "vr": 0.0, "omega": 0.0,
        } for i in range(1000)]
        blob, events = encode_compact_trajectory(samples)

        self.assertEqual(len(blob), COMPACT_HEADER_SIZE + 1000 * COMPACT_RECORD_SIZE)
        self.assertEqual(events, ())
        self.assertEqual(len(blob), 24004)

    def test_missing_field_is_rejected(self):
        samples = _samples()
        del samples[1]["omega"]
        with self.assertRaises(ValueError):
            encode_compact_trajectory(samples)

    def test_non_finite_is_rejected(self):
        samples = _samples()
        samples[1]["x"] = float("nan")
        with self.assertRaises(ValueError):
            encode_compact_trajectory(samples)

    def test_float32_overflow_is_rejected(self):
        samples = _samples()
        samples[1]["x"] = 1e40
        with self.assertRaises(ValueError):
            encode_compact_trajectory(samples)


if __name__ == "__main__":
    unittest.main()
