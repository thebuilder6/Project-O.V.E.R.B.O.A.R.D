"""
Tests for finding a Pybricks hub over Bluetooth.

The hub cannot be reached from a test, so the scanner and the connection are
faked. What is checked here is the rule that makes the host work on
Windows: the hub is found by probing what is nearby, the way the Pybricks IDE
finds it, instead of by demanding that the advertisement already carries the
Pybricks service UUID and a local name.
"""

import sys
import types
import unittest
from unittest.mock import patch

import overboard.deploy.hub_ble as hub_ble
import overboard.deploy.hub_deploy as hub_deploy


class _Adv:
    """A stand-in for bleak's advertisement data."""

    def __init__(self, local_name=None, service_uuids=()):
        self.local_name = local_name
        self.service_uuids = list(service_uuids)


class _Device:
    """A stand-in for bleak's device."""

    def __init__(self, address, name=None):
        self.address = address
        self.name = name


def _device(address, name=None, advertises=False, local_name=None):
    uuids = [hub_ble.PYBRICKS_SERVICE_UUID] if advertises else []
    return _Device(address, name), _Adv(local_name or name, uuids)


class TestFindHub(unittest.TestCase):
    """The hub is found the way the IDE finds it."""

    def _run(self, devices, name=None, hub_address=None):
        """Find a hub among the devices; return it and the probed ones."""
        probed = []

        async def probe(device, *args, **kwargs):
            probed.append(device.address)
            return device.address == hub_address

        with patch.object(hub_ble, "scan_devices",
                          _fake_scan(devices)), \
                patch.object(hub_ble, "_RELEASE_SECONDS", 0):
            device = _run_async(hub_ble.find_hub(name, probe=probe))
        return device, probed

    def test_a_device_that_advertises_the_service_is_used(self):
        devices = [_device("11:22:33:44:55:66", advertises=True)]
        device, probed = self._run(devices)
        self.assertEqual(device.address, "11:22:33:44:55:66")
        # It says what it is, so it does not have to be probed.
        self.assertEqual(probed, [])

    def test_a_device_that_only_answers_when_probed_is_used(self):
        """Windows does not always report the advertised UUID."""
        devices = [_device("11:22:33:44:55:66"),
                   _device("22:33:44:55:66:77")]
        device, probed = self._run(devices, hub_address="22:33:44:55:66:77")
        self.assertEqual(device.address, "22:33:44:55:66:77")
        # Both were asked: the hub does not have to announce itself.
        self.assertEqual(sorted(probed), ["11:22:33:44:55:66",
                                          "22:33:44:55:66:77"])

    def test_the_named_device_is_asked_first(self):
        """Two hubs nearby: the operator's choice decides."""
        devices = [_device("11:22:33:44:55:66", name="Pybricks"),
                   _device("22:33:44:55:66:77", name="Spare hub")]
        device, probed = self._run(devices, name="spare hub",
                                   hub_address="22:33:44:55:66:77")
        self.assertEqual(device.address, "22:33:44:55:66:77")
        self.assertEqual(probed[0], "22:33:44:55:66:77")

    def test_the_named_device_wins(self):
        devices = [_device("11:22:33:44:55:66", name="Pybricks"),
                   _device("22:33:44:55:66:77", name="Other hub")]
        device, _ = self._run(devices, name="other hub",
                              hub_address="22:33:44:55:66:77")
        self.assertEqual(device.address, "22:33:44:55:66:77")

    def test_substring_match_is_probed_before_unnamed_uuid_match(self):
        """KI-046.2: A device whose name is a substring match of --hub-name
        is probed before falling back to an unrelated UUID match."""
        devices = [_device("11:22:33:44:55:66", name="Pybricks", advertises=True),
                   _device("22:33:44:55:66:77", name="Team Robot 42", advertises=False)]
        device, probed = self._run(devices, name="Robot",
                                   hub_address="22:33:44:55:66:77")
        self.assertEqual(device.address, "22:33:44:55:66:77")
        self.assertIn("22:33:44:55:66:77", probed)

    def test_an_address_is_a_name_too(self):
        devices = [_device("11:22:33:44:55:66", name="Pybricks")]
        device, _ = self._run(devices, name="11:22:33:44:55:66",
                              hub_address="11:22:33:44:55:66")
        self.assertEqual(device.address, "11:22:33:44:55:66")

    def test_nothing_answered_is_reported(self):
        devices = [_device("11:22:33:44:55:66", name="Speaker")]
        with patch.object(hub_ble, "scan_devices", _fake_scan(devices)):
            with self.assertRaises(hub_ble.HubNotFound) as caught:
                _run_async(hub_ble.find_hub())
        message = str(caught.exception)
        self.assertIn("No Pybricks hub found", message)
        # The operator can see what was around, so the message is not a riddle.
        self.assertIn("Speaker (11:22:33:44:55:66)", message)
        self.assertIn("Close Pybricks Code", message)

    def test_an_empty_scan_is_reported(self):
        with patch.object(hub_ble, "scan_devices", _fake_scan([])):
            with self.assertRaises(hub_ble.HubNotFound):
                _run_async(hub_ble.find_hub())


class TestScan(unittest.TestCase):
    """The scan itself is unfiltered, and it reads both bleak shapes."""

    def test_the_scan_does_not_filter_on_the_service(self):
        """A filtered scan is what fails on Windows."""
        recorded = {}

        class _Scanner:
            @staticmethod
            async def discover(timeout=5.0, return_adv=False, **kwargs):
                recorded.update(kwargs)
                recorded["timeout"] = timeout
                return {}

        fake = types.ModuleType("bleak")
        fake.BleakScanner = _Scanner
        with patch.dict(sys.modules, {"bleak": fake}):
            _run_async(hub_ble.scan_devices(seconds=3.0))
        self.assertNotIn("service_uuids", recorded)

    def test_both_bleak_shapes_are_read(self):
        device, advertisement = _device("11:22:33:44:55:66", name="Pybricks")

        class _Scanner:
            @staticmethod
            async def discover(timeout=5.0, return_adv=False, **kwargs):
                return {"11:22:33:44:55:66": (device, advertisement)}

        fake = types.ModuleType("bleak")
        fake.BleakScanner = _Scanner
        with patch.dict(sys.modules, {"bleak": fake}):
            found = _run_async(hub_ble.scan_devices(seconds=0.01))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0].address, "11:22:33:44:55:66")


class TestHubDeployUsesTheProbe(unittest.TestCase):
    """The deployer connects to the hub the discovery found."""

    def test_the_hub_is_found_by_probing(self):
        device, _advertisement = _device("11:22:33:44:55:66", name="Pybricks")
        found = {}

        async def find_hub(name=None):
            found["name"] = name
            return device

        fake_pybricks = types.ModuleType("pybricksdev.connections.pybricks")
        fake_pybricks.PybricksHubBLE = lambda dev: ("ble", dev)
        fake_pybricks.PybricksHubUSB = lambda dev: ("usb", dev)

        with patch.dict(sys.modules, _fake_pybricksdev(fake_pybricks)), \
                patch.object(hub_ble, "find_hub", find_hub):
            hub = _run_async(hub_deploy._connect("ble", "Pybricks"))

        self.assertEqual(hub[0], "ble")
        self.assertEqual(hub[1], device)
        self.assertEqual(found["name"], "Pybricks")


class TestTheConnectionFailure(unittest.TestCase):
    """A hub that will not take the connection is an error to act on."""

    def test_it_says_the_ide_holds_the_connection(self):
        class _Hub:
            async def connect(self):
                raise OSError("device is already connected")

        async def fake_connect(connection, name):
            return _Hub()

        fake_pybricks = types.ModuleType("pybricksdev.connections.pybricks")
        fake_pybricks.HubDisconnectError = RuntimeError

        with patch.dict(sys.modules, _fake_pybricksdev(fake_pybricks)), \
                patch.object(hub_deploy, "_connect", fake_connect):
            with self.assertRaises(RuntimeError) as caught:
                hub_deploy.deploy("ignored.py")

        message = str(caught.exception)
        self.assertIn("would not take the connection", message)
        # The usual cause is the IDE holding the hub's one connection.
        self.assertIn("Close Pybricks Code", message)


class TestListHubs(unittest.TestCase):
    """The device picker is the IDE's picker."""

    def test_it_says_which_device_is_a_hub(self):
        import io
        from contextlib import redirect_stdout

        devices = [_device("11:22:33:44:55:66", name="Pybricks",
                           advertises=True),
                   _device("22:33:44:55:66:77", name="Speaker")]

        async def fake_list(seconds=10.0, timeout=5.0):
            for device, advertisement in devices:
                is_hub = hub_ble.advertises_pybricks(advertisement)
                yield device, advertisement, is_hub

        buffer = io.StringIO()
        with patch.object(hub_ble, "list_hubs", fake_list), \
                redirect_stdout(buffer):
            status = hub_deploy.list_hubs()
        output = buffer.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("Pybricks (11:22:33:44:55:66)", output)
        self.assertIn("<- Pybricks hub", output)
        self.assertIn("Speaker (22:33:44:55:66:77)", output)

    def test_no_hub_is_a_failure(self):
        import io
        from contextlib import redirect_stdout

        devices = [_device("22:33:44:55:66:77", name="Speaker")]

        async def fake_list(seconds=10.0, timeout=5.0):
            for device, advertisement in devices:
                yield device, advertisement, False

        buffer = io.StringIO()
        with patch.object(hub_ble, "list_hubs", fake_list), \
                redirect_stdout(buffer):
            status = hub_deploy.list_hubs()
        self.assertEqual(status, 1)
        self.assertIn("No Pybricks hub answered", buffer.getvalue())


class TestTheStudioHeadline(unittest.TestCase):
    """The studio shows the operator the headline, not the last line."""

    def test_the_error_line_is_the_headline(self):
        import overboard.gui_server as gui_server

        lines = [
            "error: No Pybricks hub found over Bluetooth.",
            "  - Turn the hub on.",
            "  unnamed device (11:22:33:44:55:66)",
        ]
        headline = gui_server._error_headline(lines)
        self.assertEqual(headline, "No Pybricks hub found over Bluetooth.")

    def test_a_tool_that_says_nothing_uses_the_last_line(self):
        import overboard.gui_server as gui_server

        self.assertEqual(gui_server._error_headline(["", "boom"]), "boom")
        self.assertEqual(gui_server._error_headline([]), "The push failed.")


def _fake_pybricksdev(pybricks_module):
    """The pybricksdev modules a test needs, so the real one is not needed."""
    return {
        "pybricksdev": types.ModuleType("pybricksdev"),
        "pybricksdev.connections": types.ModuleType("pybricksdev.connections"),
        "pybricksdev.connections.pybricks": pybricks_module,
    }


def _fake_scan(devices):
    """A stand-in for hub_ble.scan_devices."""
    async def scan(seconds=10.0):
        return list(devices)
    return scan


def _run_async(coroutine):
    """Run a coroutine to the end."""
    import asyncio
    return asyncio.run(coroutine)


if __name__ == "__main__":
    unittest.main()
