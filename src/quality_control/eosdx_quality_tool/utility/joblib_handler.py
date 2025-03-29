def compute_statistics(obj):
    """
    Compute basic statistics for the loaded joblib object.
    """
    stats = "File Statistics:\n"
    stats += f"Type: {type(obj).__name__}\n"

    # Attempt to report length if applicable
    try:
        stats += f"Length: {len(obj)}\n"
    except TypeError:
        stats += "Length: Not applicable\n"

    # If the object has a shape attribute (e.g., numpy arrays), display it
    if hasattr(obj, "shape"):
        stats += f"Shape: {obj.shape}\n"

    if hasattr(obj, "columns"):
        stats += f"Shape: {obj.columns}\n"

    return stats
