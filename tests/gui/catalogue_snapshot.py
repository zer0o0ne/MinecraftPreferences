"""The complete GUI catalogue relocated byte-for-byte from the approved config."""

from pathlib import Path

import yaml

from mcpref.gui.screens import load_screens

PATH = Path(__file__).with_name("catalogue_snapshot.yaml")
TEXT = PATH.read_text()


def classes() -> dict:
    return yaml.load(TEXT, Loader=yaml.CSafeLoader)["classes"]


def screens() -> dict:
    return load_screens(PATH)
