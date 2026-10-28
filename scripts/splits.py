import hashlib

# Every model should use this so results are compared on the same cognate sets.
# Splitting by cognate set keeps related forms from leaking between train and test.


def assign_split(cognateset_id):
    """Deterministically assign a cognate set to train (80%), val (10%), or test (10%)."""
    bucket = int(hashlib.md5(str(cognateset_id).encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 8:
        return "val"
    if bucket == 9:
        return "test"
    return "train"
