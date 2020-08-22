class Itemset(set):

    def __init__(self, iterable, occurrences=None):
        super().__init__(iterable)
        self.occurrences = occurrences

    def __hash__(self):
        return hash(tuple(sorted(self)))

    def compute_occurrences(self, sdb):
        iter_itemset = iter(self)
        item = next(iter_itemset)
        self.occurrences = {
            i: (s.index(item), len(s) - s[::-1].index(item) - 1)
            for i, s in enumerate(sdb)
            if item in s
        }
        for item in iter_itemset:
            self.occurrences = self.update_occurrences(item, sdb)

    def update_occurrences(self, c, sdb):
        return {
            sid: (
                max(first_occurrence, sdb[sid].index(c)),
                min(last_occurrence, len(sdb[sid]) - sdb[sid][::-1].index(c) - 1)
            )
            for sid, (first_occurrence, last_occurrence) in self.occurrences.items()
            if c in sdb[sid]
        }
