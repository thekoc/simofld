def singular_not_none(*args):
    """Whether only one of the arguments has a value (not None)

    Returns:
        bool: Only one of the arguments has a value (not None)
    """
    has_value = False
    for arg in args:
        if arg is not None:
            if has_value:
                return False
            has_value = True
    return has_value