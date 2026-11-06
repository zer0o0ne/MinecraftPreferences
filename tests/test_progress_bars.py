"""Parser checks for tqdm terminal control sequences."""

from progress_bars import Bar, assert_progress, parse, states


def test_cursor_up_after_reference_states_is_ignored():
    values = [
        "pretrain reference:   0%|          | 0/2 [00:00<?, ?pair/s]",
        "pretrain reference:  50%|#####     | 1/2 [00:00<00:00, 2.00pair/s]",
        "pretrain reference: 100%|##########| 2/2 [00:00<00:00, 2.00pair/s]",
    ]
    err = "\r" + "\r".join(f"{value}\x1b[A" for value in values) + "\n"
    expected = [Bar("pretrain reference", i, 2, "pair") for i in range(3)]
    assert parse(values[0] + "\x1b[A") == expected[0]
    assert parse(values[0] + "\x1b[A\x1b[A") == expected[0]
    assert parse(
        "pretrain gradient flow collect: 100%|##########| 2/2 [00:00<00:00, 2.00micro-batch/s]"
    ) == Bar("pretrain gradient flow collect", 2, 2, "micro-batch")
    assert states(err, "pretrain reference") == expected
    assert_progress(err)


def test_nested_reference_closes_before_outer_pretrain():
    inner = [
        "pretrain reference:   0%|          | 0/2 [00:00<?, ?pair/s]",
        "pretrain reference:  50%|#####     | 1/2 [00:00<00:00, 2.00pair/s]",
        "pretrain reference: 100%|##########| 2/2 [00:00<00:00, 2.00pair/s]",
    ]
    states_text = [
        "pretrain:   0%|          | 0/6 [00:00<?, ?step/s]",
        "pretrain:  67%|######6   | 4/6 [00:00<00:00, 2.00step/s]\n",
        *(f"{value}\x1b[A" for value in inner[:-1]),
        f"{inner[-1]}\x1b[A\n",
        "pretrain:  83%|########3 | 5/6 [00:00<00:00, 2.00step/s]",
        "pretrain: 100%|##########| 6/6 [00:00<00:00, 2.00step/s]",
    ]
    assert_progress("\r" + "\r".join(states_text) + "\n")


def test_repeated_bars_have_independent_totals_and_close_states():
    values = [
        "work:   0%|          | 0/2 [00:00<?, ?item/s]",
        "work:  50%|#####     | 1/2 [00:00<00:00, 2.00item/s]",
        "work: 100%|##########| 2/2 [00:00<00:00, 2.00item/s]\n",
        "work:   0%|          | 0/3 [00:00<?, ?item/s]",
        "work:  33%|###3      | 1/3 [00:00<00:00, 2.00item/s]",
        "work:  67%|######6   | 2/3 [00:00<00:00, 2.00item/s]",
        "work: 100%|##########| 3/3 [00:00<00:00, 2.00item/s]",
    ]
    assert_progress("\r" + "\r".join(values) + "\n")
