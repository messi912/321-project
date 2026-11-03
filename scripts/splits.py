import hashlib

# Every model should use this so results are compared on the same cognate sets.
# Splitting by root (24773 from 24773_4) keeps every set from one root in the same split,
# so related forms like "die" make and "kill" hoʔomake can't leak between train and test.


def assign_split(cognateset_id):
    """Deterministically assign a cognate set to train (80%), val (10%), or test (10%) by its root."""
    root = str(cognateset_id).split("_")[0]
    bucket = int(hashlib.md5(root.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 8:
        return "val"
    if bucket == 9:
        return "test"
    return "train"
