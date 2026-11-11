import hashlib

# Every model should use this so results are compared on the same cognate sets.
# Splitting by cognate set keeps related forms from leaking between train and test.


def assign_split(cognateset_id):
    """Assign all cognate sets sharing a root to the same data split."""

    root_id = str(cognateset_id).split("_")[0]

    bucket = int(hashlib.md5(root_id.encode("utf-8")).hexdigest(), 16) % 10

    if bucket == 8:
        return "val"
    if bucket == 9:
        return "test"
    return "train"

