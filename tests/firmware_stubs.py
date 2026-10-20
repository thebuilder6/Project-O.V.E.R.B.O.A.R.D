"""
Stubs that make the Pybricks firmware importable on the host.

The firmware modules import ``pybricks``, ``umath``, ``ustruct`` and
``micropython``, none of which exist on CPython. The stubs below stand in for
them so the real firmware code can be executed and checked on the host.

The installer is not destructive: it fills in what is missing and leaves
anything another test already installed alone. That way two test modules can
use it in the same process, in either order.
"""

import asyncio
import math
import struct
import sys
import types


def install_firmware_stubs():
    """Make the firmware modules importable on the host."""
    sys.modules.setdefault("umath", math)
    sys.modules.setdefault("ustruct", struct)

    if "micropython" not in sys.modules:
        micropython = types.ModuleType("micropython")
        micropython.const = lambda value: value
        sys.modules["micropython"] = micropython

    pybricks = sys.modules.get("pybricks")
    if pybricks is None:
        pybricks = types.ModuleType("pybricks")
        sys.modules["pybricks"] = pybricks

    class StopWatch:
        def __init__(self):
            self._start = None

        def time(self):
            import time
            if self._start is None:
                self._start = time.monotonic()
            return int((time.monotonic() - self._start) * 1000.0)

    async def wait(_ms):
        # Sleep for real so a loop advances at roughly the deployed rate
        # instead of spinning on the host.
        await asyncio.sleep(min(_ms, 20) / 1000.0)

    def run_task(coro):
        coro.close()
        return True

    tools = sys.modules.get("pybricks.tools")
    if tools is None:
        tools = types.ModuleType("pybricks.tools")
        sys.modules["pybricks.tools"] = tools
    if not hasattr(tools, "StopWatch"):
        tools.StopWatch = StopWatch
    if not hasattr(tools, "wait"):
        tools.wait = wait
    if not hasattr(tools, "run_task"):
        tools.run_task = run_task
    pybricks.tools = tools

    class Button:
        CENTER = "CENTER"
        BLUETOOTH = "BLUETOOTH"

    class Color:
        YELLOW = "YELLOW"
        RED = "RED"
        BLUE = "BLUE"
        GREEN = "GREEN"

    parameters = sys.modules.get("pybricks.parameters")
    if parameters is None:
        parameters = types.ModuleType("pybricks.parameters")
        sys.modules["pybricks.parameters"] = parameters
    if not hasattr(parameters, "Button"):
        parameters.Button = Button
    if not hasattr(parameters, "Color"):
        parameters.Color = Color
    pybricks.parameters = parameters


class GcStub:
    """MicroPython's gc has mem_free() and threshold(); CPython's does not."""

    @staticmethod
    def mem_free():
        return 100000

    @staticmethod
    def collect():
        pass

    @staticmethod
    def threshold(_value):
        pass
