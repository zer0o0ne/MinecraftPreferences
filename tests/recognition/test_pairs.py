"""`VARIANTS` of src/mcpref/recognition/pairs.py against specs/gui-masks.md (Interface); the
pairs it builds are checked through `main` in test_evaluate_outputs.py."""

from mcpref.recognition.pairs import VARIANTS


def test_variants_in_order():
    """The variant names a ... f, in this order (Notation)."""
    assert list(VARIANTS) == ["a", "b", "c", "d", "e", "f"]
