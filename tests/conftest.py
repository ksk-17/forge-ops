import sys
from pathlib import Path

# Add project root to Python path so tests can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture
def fresh_bus():
    """Install an isolated EventBus as the global bus for one test."""
    from forge_events import EventBus, get_bus, set_bus

    previous = get_bus()
    bus = EventBus()
    set_bus(bus)
    yield bus
    set_bus(previous)
