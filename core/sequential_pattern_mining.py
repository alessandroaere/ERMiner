import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm


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
                min(first_occurrence, sdb[sid].index(c)),
                max(last_occurrence, len(sdb[sid]) - sdb[sid][::-1].index(c) - 1)
            )
            for sid, (first_occurrence, last_occurrence) in self.occurrences.items()
            if c in sdb[sid]
        }


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


class ERMiner:

    @staticmethod
    def cooccurs(a, b, sdb):
        return sum(a in s and b in s for s in sdb) / len(sdb)

    def __init__(self, minsup, minconf, single_consequent=False):
        self.minsup = minsup
        self.minconf = minconf
        self.single_consequent = single_consequent
        self._left_store = dict()
        self.valid_rules = None
        self._SCM = None

    def _find_left_equivalence_classes(self, i, rules, sdb):
        return {
            W: {rule for rule in rules if rule.antecedent == W and len(rule.consequent) == i}
            for W in {rule.antecedent for rule in rules if rule.is_frequent(sdb, self.minsup)}
        }

    def _find_right_equivalence_classes(self, i, rules, sdb):
        return {
            W: {rule for rule in rules if rule.consequent == W and len(rule.antecedent) == i}
            for W in {rule.consequent for rule in rules if rule.is_frequent(sdb, self.minsup)}
        }

    def _first_scan(self, sdb):
        itemset = {i for s in sdb for i in s}
        self._SCM = {
            tuple(sorted((a, b))): self.cooccurs(a, b, sdb)
            for a, b in itertools.combinations(itemset, 2)
        }
        rules11 = {
            Rule(Itemset([a]), Itemset([c])) for a, c in itertools.product(itemset, repeat=2)
        }
        frequent_rules11 = {r for r in tqdm(rules11) if r.is_frequent(sdb, self.minsup)}
        self.valid_rules = {
            r for r in frequent_rules11 if r.is_valid(sdb, self.minsup, self.minconf)
        }
        leq = self._find_left_equivalence_classes(1, frequent_rules11, sdb)
        req = self._find_right_equivalence_classes(1, frequent_rules11, sdb)
        return leq, req

    def _left_search(self, leq, sdb):
        leq1 = set()
        for r, s in itertools.combinations(leq, 2):
            yr = sorted(r.consequent)
            ys = sorted(s.consequent)
            if yr[:-1] == ys[:-1]:
                c = yr[-1]
                d = ys[-1]
                if self._SCM[tuple(sorted({c, d}))] >= self.minsup:
                    yrud = Itemset(yr + [d], r.consequent.update_occurrences(d, sdb))
                    t = Rule(r.antecedent, yrud)
                    if t.is_frequent(sdb, self.minsup):
                        leq1.add(t)
                        if t.is_valid(sdb, self.minsup, self.minconf):
                            self.valid_rules.add(t)
        if leq1:
            self._left_search(leq1, sdb)

    def _right_search(self, req, sdb):
        req1 = set()
        for r, s in itertools.combinations(req, 2):
            xr = sorted(r.antecedent)
            xs = sorted(s.antecedent)
            if xr[:-1] == xs[:-1]:
                c = xr[-1]
                d = xs[-1]
                if self._SCM[tuple(sorted({c, d}))] >= self.minsup:
                    xrud = Itemset(xr + [d], r.antecedent.update_occurrences(d, sdb))
                    t = Rule(xrud, r.consequent)
                    if t.is_frequent(sdb, self.minsup):
                        req1.add(t)
                        if t.antecedent not in self._left_store:
                            self._left_store[t.antecedent] = set()
                        self._left_store[t.antecedent].add(t)
                        if t.is_valid(sdb, self.minsup, self.minconf):
                            self.valid_rules.add(t)
        if req1:
            self._right_search(req1, sdb)

    def fit(self, sdb):
        leq, req = self._first_scan(sdb)
        if not self.single_consequent:
            for H in tqdm(leq.values()):
                self._left_search(H, sdb)
        for J in tqdm(req.values()):
            self._right_search(J, sdb)
        if not self.single_consequent:
            for K in tqdm(self._left_store.values()):
                self._left_search(K, sdb)

    def rules_to_df(self, csv_file):
        df = pd.DataFrame(
            [
                [list(r.antecedent), list(r.consequent), r.support, r.confidence]
                for r in self.valid_rules
            ],
            columns=['antecedent', 'consequent', 'support', 'confidence']
        )
        df = df.sort_values(by=['confidence', 'support'], ascending=False)
        df.to_csv(csv_file, index=False)
