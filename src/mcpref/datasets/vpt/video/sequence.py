"""Index views preserve bounded readers when selecting VPT rows."""

from collections.abc import Sequence


class RowView(Sequence):
    def __init__(self, rows: Sequence, indices: Sequence):
        self.rows, self.indices = rows, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return RowView(self.rows, self.indices[index])
        return self.rows[self.indices[index]]
