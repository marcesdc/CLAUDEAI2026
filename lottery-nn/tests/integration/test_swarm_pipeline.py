"""Integration tests for the multi-lottery swarm pipeline -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


def test_swarm_build_all_features(swarm_features_all):
    pass


def test_swarm_forward_all_lottery_ids(dummy_swarm_batch):
    pass
