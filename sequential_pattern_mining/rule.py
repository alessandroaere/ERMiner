class Rule:

    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
        self.support = None
        self.confidence = None
        self.sequences_with_rule = None

    def __str__(self):
        return '{} -> {}'.format(self.antecedent, self.consequent)

    def __hash__(self):
        return hash((self.antecedent, self.consequent))

    def find_sequences_with_rule(self, sdb):
        if self.antecedent.occurrences is None:
            self.antecedent.compute_occurrences(sdb)
        if self.consequent.occurrences is None:
            self.consequent.compute_occurrences(sdb)
        self.sequences_with_rule = [
            k
            for k, v in self.antecedent.occurrences.items()
            if k in self.consequent.occurrences.keys() and v[0] < self.consequent.occurrences[k][1]
        ]

    def compute_support(self, sdb):
        if self.sequences_with_rule is None:
            self.find_sequences_with_rule(sdb)
        self.support = len(self.sequences_with_rule) / len(sdb)

    def compute_confidence(self, sdb):
        if self.support is None:
            self.compute_support(sdb)
        self.confidence = self.support * len(sdb) / len(self.antecedent.occurrences)

    def is_frequent(self, sdb, minsup):
        if self.support is None:
            self.compute_support(sdb)
        return self.support >= minsup

    def is_valid(self, sdb, minsup, minconf):
        if not self.is_frequent(sdb, minsup):
            return False
        if self.confidence is None:
            self.compute_confidence(sdb)
        return self.confidence >= minconf
