import pytest
from analyzer import validate_asset_name

def test_valid_asset_names():
    # Standard assertion: expected output is True
    assert validate_asset_name("BTC") is True
    assert validate_asset_name("Ethereum") is True

def test_invalid_short_names():
    # Edge case assertion: should return False for names too short
    assert validate_asset_name("X") is False
    assert validate_asset_name("") is False
