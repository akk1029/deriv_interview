def validate_asset_name(name: str) -> bool:
    """Returns True if the asset naming standard matches basic rules."""
    if not name or len(name) < 3:
        return False
    return True
