"""Tests for scripts/data.py: turning words into numbers for the model and back."""
from conftest import ACD_TABLE, need
from data import Vocab, decode, encode, load_examples

EXAMPLE = {
    "cognateset_id": "24773_4",
    "target_language": "hawaiian",
    "target": "make",
    "sources": {"maori": "mate", "samoan": "mate", "tongan": "mate"},
}


def test_encode_then_decode_gives_the_word_back():
    vocab = Vocab.from_examples([EXAMPLE])
    _, target_ids = encode(EXAMPLE, vocab)
    assert decode(target_ids[1:], vocab) == "make"     # skip the <start> token


def test_source_starts_with_the_target_language_tag():
    vocab = Vocab.from_examples([EXAMPLE])
    source_ids, _ = encode(EXAMPLE, vocab)
    assert vocab.token(source_ids[0]) == "<to:hawaiian>"
    assert vocab.token(source_ids[1]) == "<maori>"


def test_target_is_wrapped_in_start_and_end():
    vocab = Vocab.from_examples([EXAMPLE])
    _, target_ids = encode(EXAMPLE, vocab)
    assert target_ids[0] == vocab.start_id
    assert target_ids[-1] == vocab.end_id


def test_unseen_letters_become_unknown():
    vocab = Vocab.from_examples([EXAMPLE])
    assert vocab.lookup("ŋ") == vocab.unk_id


def test_the_hidden_word_is_never_a_source():
    # The model must never see the answer as one of its inputs
    examples = load_examples(need(ACD_TABLE))
    for split in examples.values():
        for example in split:
            assert example["target_language"] not in example["sources"]


def test_test_words_are_not_in_training_examples():
    examples = load_examples(need(ACD_TABLE))
    train_sets = {e["cognateset_id"] for e in examples["train"]}
    test_sets = {e["cognateset_id"] for e in examples["test"]}
    assert not train_sets & test_sets
