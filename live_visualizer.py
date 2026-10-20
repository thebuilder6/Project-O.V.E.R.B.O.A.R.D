import asyncio
import json
import socket
import threading
import websockets
from typing import List, Dict, Any, Optional

class LiveVisualizer:
    """
    WebSocket server for live trajectory visualization.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.thread = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        # Replayed to every client that connects after the solve has started
        # (KI-016: the first client used to miss the config and waypoints).
        self.last_config: Optional[Dict[str, Any]] = None
        self.last_trajectory: Optional[Dict[str, Any]] = None

    def start(self):
        """Start the WebSocket server in a separate thread.

        The port is checked first so a busy port gets a clear error instead of
        killing whatever owns it (KI-005).
        """
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((self.host, self.port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {self.port} is already in use. "
                f"Pick another with --port or stop the other program."
            ) from exc
        finally:
            probe.close()

        print(f"Live visualizer: Initializing on {self.host}:{self.port}...")
        self.thread = threading.Thread(target=self._run_server, daemon=True, name="VisualizerThread")
        self.thread.start()

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._main())
        except Exception as e:
            print(f"Live visualizer: WebSocket server error: {e}")
        finally:
            self.loop.close()

    async def _main(self):
        try:
            async with websockets.serve(self._handler, self.host, self.port):
                print(f"Live visualizer: Server active at ws://{self.host}:{self.port}")
                while not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Live visualizer: Failed to start server: {e}")

    async def _handler(self, websocket):
        # Ensure thread-safe modification of clients set
        with self._lock:
            self.clients.add(websocket)
            replay = [m for m in (self.last_config, self.last_trajectory) if m]

        try:
            for message in replay:
                await websocket.send(json.dumps(message))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
        finally:
            with self._lock:
                self.clients.discard(websocket)

    def broadcast(self, data: Dict[str, Any]):
        """Send a message to all connected clients."""
        if not self.loop or not self.clients:
            return
            
        message = json.dumps(data)
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)
        except RuntimeError:
            pass

    def send_trajectory(self, trajectory: List[Dict[str, Any]], phase: str = "solve", iteration: int = 0):
        """Send current trajectory state."""
        message = {
            "type": "trajectory",
            "phase": phase,
            "iteration": iteration,
            "samples": trajectory
        }
        self.last_trajectory = message
        self.broadcast(message)

    def send_config(self, robot_config: Dict[str, Any], waypoints: List[Any]):
        """Send initial configuration."""
        message = {
            "type": "config",
            "robot": robot_config,
            "waypoints": waypoints
        }
        self.last_config = message
        self.broadcast(message)

    async def _broadcast(self, message):
        # Snapshoting clients to avoid modification issues
        current_clients = []
        with self._lock:
            current_clients = list(self.clients)
            
        if current_clients:
            # Send to all clients, ignore failures for individual clients
            for client in current_clients:
                try:
                    await client.send(message)
                except:
                    pass

    def stop(self):
        """Stop the server."""
        print("Live visualizer: Stopping...")
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

# Global instance for easy access
_global_visualizer = None

def get_visualizer(port: int = 8765, host: str = "127.0.0.1"):
    """Return the shared visualizer, starting it on first use."""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = LiveVisualizer(host=host, port=port)
        _global_visualizer.start()
    return _global_visualizer
