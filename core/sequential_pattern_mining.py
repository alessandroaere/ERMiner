import itertools
import pandas as pd


class Rule:

    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
        self.support = None
        self.confidence = None

    def __str__(self):
        return '{} -> {}; sup: {}; conf: {}'.format(self.antecedent,
                                                    self.consequent,
                                                    self.support,
                                                    self.confidence)

    def __hash__(self):
        return hash(self.antecedent.__str__() + self.consequent.__str__())

    def __eq__(self, other):
        return self.antecedent == other.antecedent and self.consequent == other.consequent

    def occurs(self, sequence):
        return any(self.antecedent.occurs(sequence[:i]) and self.consequent.occurs(sequence[i:])
                   for i in range(len(sequence)))

    def get_sup(self, sequences):
        self.support = sum(self.occurs(s) for s in sequences) / len(sequences)
        return self.support

    def get_conf(self, sequences):
        self.confidence = sum(self.occurs(s) for s in sequences) / \
                          sum(self.antecedent.occurs(s) for s in sequences)
        return self.confidence

    def is_frequent(self, sequences, minsup):
        self.get_sup(sequences)
        return self.support >= minsup

    def is_valid(self, sequences, minsup, minconf):
        self.get_conf(sequences)
        return self.is_frequent(sequences, minsup) and self.confidence >= minconf


class Itemset:

    def __init__(self, obj):
        self.elements = obj

    def __str__(self):
        return self.elements.__str__()

    def __hash__(self):
        return hash(frozenset(self.elements))

    def __eq__(self, other):
        return frozenset(self.elements) == frozenset(other.elements)

    def occurs(self, sequence):
        return all(i in sequence for i in self.elements)

    def size(self):
        return len(self.elements)


class RulesDatabase:

    def __init__(self, rules):
        self.rules = rules

    def __add__(self, other):
        self.rules.add(other)
        return self

    def left_equivalence(self, W, i=None):
        if i:
            return set([Rule(r.antecedent, r.consequent) for r in self.rules
                        if W == r.antecedent and r.consequent.size() == i])
        else:
            return set([Rule(r.antecedent, r.consequent) for r in self.rules if W == r.antecedent])

    def right_equivalence(self, W, i=None):
        if i:
            return set([Rule(r.antecedent, r.consequent) for r in self.rules
                        if W == r.consequent and r.antecedent.size() == i])
        else:
            return set([Rule(r.antecedent, r.consequent) for r in self.rules if W == r.consequent])

    def equivalence_classes(self, i=None, j=None):
        LE_classes = {a: self.left_equivalence(a, i) for a in {r.antecedent for r in self.rules}}
        RE_classes = {c: self.right_equivalence(c, j) for c in {r.consequent for r in self.rules}}
        return LE_classes, RE_classes


class ERMiner:

    @staticmethod
    def cooccurs(a, b, sequences):
        return sum(a in s and b in s for s in sequences) / len(sequences)

    def __init__(self, minsup, minconf, single_consequent=False):
        self.minsup = minsup
        self.minconf = minconf
        self.single_consequent = single_consequent
        self.left_store = set()
        self.rules = None
        self.SCM = None

    def _first_scan(self, SBD):
        itemset = {i for s in SBD for i in s}
        self.SCM = {frozenset({a, b}): self.cooccurs(a, b, SBD)
                    for a, b in itertools.combinations(itemset, 2)}
        rules11 = {Rule(Itemset([a]), Itemset([c])) for a in itemset for c in itemset}
        frequent_rules = RulesDatabase({r for r in rules11 if r.is_frequent(SBD, self.minsup)})
        self.rules = RulesDatabase({r for r in frequent_rules.rules
                                    if r.is_valid(SBD, self.minsup, self.minconf)})
        return frequent_rules.equivalence_classes(1, 1)

    def _left_search(self, LE, SBD):
        LE1 = LE[0], set()
        for r, s in itertools.combinations(LE[1], 2):
            c = list({*r.consequent.elements} - {*s.consequent.elements})[0]
            d = list({*s.consequent.elements} - {*r.consequent.elements})[0]
            if self.SCM[frozenset({c, d})] >= self.minsup and \
                    r.consequent.elements[:-1] == s.consequent.elements[:-1]:
                t = Rule(LE[0], Itemset(list({*r.consequent.elements, *s.consequent.elements})))
                t.get_sup(SBD)
                if t.support >= self.minsup:
                    t.get_conf(SBD)
                    if t.confidence >= self.minconf:
                        self.rules += t
                    LE1[1].add(t)
        if LE1[1]:
            self._left_search(LE1, SBD)

    def _right_search(self, RE, SBD, left_store):
        RE1 = RE[0], set()
        for r, s in itertools.combinations(RE[1], 2):
            c = list({*r.antecedent.elements} - {*s.antecedent.elements})[0]
            d = list({*s.antecedent.elements} - {*r.antecedent.elements})[0]
            if self.SCM[frozenset({c, d})] >= self.minsup and \
                    r.antecedent.elements[:-1] == s.antecedent.elements[:-1]:
                t = Rule(Itemset(list({*r.antecedent.elements, *s.antecedent.elements})), RE[0])
                t.get_sup(SBD)
                if t.support >= self.minsup:
                    t.get_conf(SBD)
                    if t.confidence >= self.minconf:
                        self.rules += t
                    RE1[1].add(t)
                    self.left_store.add(t)
        if RE1[1]:
            self._right_search(RE1, SBD, left_store)

    def fit(self, SBD):
        lEQ, rEQ = self._first_scan(SBD)
        if not self.single_consequent:
            for H in lEQ.items():
                self._left_search(H, SBD)
        for J in rEQ.items():
            self._right_search(J, SBD, self.left_store)
        if not self.single_consequent:
            self.left_store = RulesDatabase(self.left_store)
            self.left_store, _ = self.left_store.equivalence_classes()
            for K in self.left_store.items():
                self._left_search(K, SBD)
        return self.rules

    def rules_to_df(self, csv_file):
        l = []
        for r in self.rules.rules:
            l.append([r.antecedent,
                      r.consequent,
                      r.support,
                      r.confidence])
        df = pd.DataFrame(l, columns=['antecedent',
                                      'consequent',
                                      'support',
                                      'confidence'])
        df = df.sort_values(by=['confidence', 'support'], ascending=False)
        df.to_csv(csv_file)
