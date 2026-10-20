"""
Find a Pybricks hub over Bluetooth the way the Pybricks IDE does.

The Pybricks IDE (code.pybricks.com) lists the Bluetooth devices the
operating system can see and probes the one you pick for the Pybricks
service. ``pybricksdev``'s own discovery does the opposite: it accepts an
advertisement only when it carries the Pybricks service UUID *and* a local
name. Windows usually reports neither in the shape that filter wants: bleak's
WinRT backend leaves ``service_uuids`` empty unless the peripheral puts the
UUID in the advertisement payload, and the local name tends to arrive in a
scan response the scanner never joins. The strict filter then never matches,
and the host cannot reach a hub the IDE reaches every day.

So this module scans broadly, prefers the devices that look like a hub, and
probes the rest. That is what the IDE does:

    python hub_deploy.py --list-hubs

The Pybricks protocol itself is still spoken by pybricksdev; this module only
decides which device to talk to.
"""

import asyncio

# The Pybricks GATT service. A device that exposes it speaks the protocol.
PYBRICKS_SERVICE_UUID = "c5f50001-8280-46da-89f4-6d8051e4aeef"

# How long to listen for advertisements, and how long a probed device has to
# accept a connection.
SCAN_SECONDS = 10.0
PROBE_TIMEOUT = 5.0


class HubNotFound(RuntimeError):
    """No device answered as a Pybricks hub.

    It is a RuntimeError, so the callers that already report a transport
    failure report this one the same way.
    """


def _text(value):
    return str(value).strip() if value else ""


def _uuid(value):
    return str(value).lower()


def _address(device):
    return _text(getattr(device, "address", ""))


def device_names(device, advertisement):
    """Every name this device is known by."""
    names = []
    for value in (getattr(device, "name", None),
                  getattr(advertisement, "local_name", None)):
        name = _text(value)
        if name and name not in names:
            names.append(name)
    return names


def describe_device(device, advertisement):
    """A short label for a device, so the operator can recognise it."""
    names = device_names(device, advertisement)
    label = names[0] if names else "unnamed device"
    address = _address(device)
    return f"{label} ({address})" if address else label


def advertises_pybricks(advertisement):
    """True if the advertisement carries the Pybricks service UUID."""
    uuids = getattr(advertisement, "service_uuids", None) or []
    return PYBRICKS_SERVICE_UUID in {_uuid(u) for u in uuids}


def _is_address(text):
    """True if the text looks like a Bluetooth address."""
    cleaned = text.replace(":", "").replace("-", "").replace(" ", "")
    return len(cleaned) == 12 and all(
        c in "0123456789ABCDEFabcdef" for c in cleaned)


def _rank(device, advertisement, wanted):
    """Order the candidates so the hub the operator meant comes first."""
    names = [name.lower() for name in device_names(device, advertisement)]
    if wanted:
        wanted_text = _text(wanted).lower()
        if _is_address(wanted_text):
            same = (_address(device).replace(":", "").lower()
                    == wanted_text.replace(":", "").lower())
            if same:
                return 0
        if wanted_text in names:
            return 1
        if any(wanted_text in name for name in names):
            return 2
    if advertises_pybricks(advertisement):
        return 3
    if any("pybricks" in name for name in names):
        return 4
    return 5


async def scan_devices(seconds=SCAN_SECONDS):
    """Return the (device, advertisement) pairs that are advertising.

    The scan is deliberately unfiltered. Filtering on the advertised service
    UUID is what fails on Windows, so the filter would have to be "everything".
    """
    from bleak import BleakScanner

    found = await BleakScanner.discover(timeout=seconds, return_adv=True)
    devices = {}
    for value in found.values():
        device, advertisement = (value if isinstance(value, tuple)
                                 else (value, value))
        devices[_address(device) or str(id(device))] = (device, advertisement)
    return list(devices.values())


async def speaks_pybricks(device, timeout=PROBE_TIMEOUT):
    """Connect to a device and check whether it has the Pybricks service."""
    from bleak import BleakClient

    try:
        async with BleakClient(device, timeout=timeout) as client:
            return any(_uuid(service.uuid) == PYBRICKS_SERVICE_UUID
                       for service in client.services)
    except Exception:
        # A device that will not let us in is not a hub we can use.
        return False


# Windows keeps the device for a moment after the connection is closed, so
# the real connection is made a beat later.
_RELEASE_SECONDS = 0.5


async def find_hub(name=None, seconds=SCAN_SECONDS, probe=None, limit=4):
    """Return the bleak device of a Pybricks hub.

    name is the hub's Bluetooth name or address. It is a hint, not a filter:
    a hub that does not report its name over the advertisement is still
    found, because Windows often does not report it.
    """
    probe = probe or speaks_pybricks
    seen = await scan_devices(seconds)
    if not seen:
        raise HubNotFound(_no_hub_message(name, []))

    ranked = sorted(
        ((_rank(device, advertisement, name), device, advertisement)
         for device, advertisement in seen),
        key=lambda row: row[0])
    # The device the operator named (including substring matches) is asked
    # first, so an unnamed UUID match does not steal the connection (KI-046.2).
    preferred = [row for row in ranked if row[0] <= 2]
    others = [row for row in ranked if row[0] > 2]

    tried = []
    for _rank_value, device, advertisement in preferred:
        if advertises_pybricks(advertisement):
            return device
        if await probe(device):
            await asyncio.sleep(_RELEASE_SECONDS)
            return device
        tried.append(describe_device(device, advertisement))

    # A device that advertises the Pybricks service is a hub: the profile
    # says so, so it does not have to be asked.
    for _rank_value, device, advertisement in others:
        if advertises_pybricks(advertisement):
            return device

    # The rest have to be asked, a few at a time: Windows does not always
    # report the advertised UUID or the name.
    hub = await _probe_all([row[1] for row in others], probe, limit)
    if hub is not None:
        await asyncio.sleep(_RELEASE_SECONDS)
        return hub

    tried += [describe_device(row[1], row[2]) for row in others]
    raise HubNotFound(_no_hub_message(name, tried))


async def list_hubs(seconds=SCAN_SECONDS, timeout=PROBE_TIMEOUT, limit=4):
    """Yield (device, advertisement, is_hub) for every device that is visible.

    This is the IDE's device picker: it shows what is around and says which
    of them is a hub.
    """
    seen = await scan_devices(seconds)
    semaphore = asyncio.Semaphore(max(1, limit))

    async def check(device, advertisement):
        await semaphore.acquire()
        try:
            is_hub = advertises_pybricks(advertisement)
            if not is_hub:
                is_hub = await speaks_pybricks(device, timeout)
            return device, advertisement, is_hub
        finally:
            semaphore.release()

    tasks = [asyncio.create_task(check(device, advertisement))
             for device, advertisement in seen]
    try:
        for finished in asyncio.as_completed(tasks):
            yield await finished
    finally:
        for task in tasks:
            task.cancel()


async def _probe_all(devices, probe, limit):
    """Ask a few devices at a time and return the first that is a hub."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def check(device):
        await semaphore.acquire()
        try:
            return device if await probe(device) else None
        finally:
            semaphore.release()

    tasks = [asyncio.create_task(check(device)) for device in devices]
    try:
        for finished in asyncio.as_completed(tasks):
            hub = await finished
            if hub is not None:
                return hub
    finally:
        for task in tasks:
            task.cancel()
    return None


def _no_hub_message(name, tried):
    """What to tell the operator when no device answered as a hub."""
    lines = [
        "No Pybricks hub found over Bluetooth.",
        "",
        "  - Turn the hub on. It only answers while it is awake.",
        "  - Close Pybricks Code (code.pybricks.com). Only one Bluetooth",
        "    central can hold the hub's single connection, and the IDE holds",
        "    it while it is open.",
        "  - Pair the hub once in the Bluetooth settings if it has never",
        "    been paired.",
    ]
    if name:
        lines.append(f"  - '{name}' did not answer either. The name is a hint,"
                     " not a filter, so every nearby device was tried.")
    if tried:
        lines.append("")
        lines.append("Tried, and none of them is a hub:")
        lines += [f"  {item}" for item in tried]
    return "\n".join(lines)
