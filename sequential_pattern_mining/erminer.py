import itertools
from typing import Sequence, Dict, Set, Tuple

import pandas as pd
from tqdm import tqdm

from .itemset import Itemset
from .rule import Rule


class ERMiner:

    @staticmethod
    def cooccurs(a: Itemset, b: Itemset, sdb: Sequence[Sequence]) -> float:
        return sum(a in s and b in s for s in sdb) / len(sdb)

    def __init__(self, minsup: float, minconf: float, single_consequent: bool = False):
        self.minsup = minsup
        self.minconf = minconf
        self.single_consequent = single_consequent
        self._left_store = dict()
        self.valid_rules = None
        self._SCM = None

    def _find_left_equivalence_classes(self,
                                       i: int,
                                       rules: Set[Rule],
                                       sdb: Sequence[Sequence]) -> Dict[Itemset, Set[Rule]]:
        return {
            W: {rule for rule in rules if rule.antecedent == W and len(rule.consequent) == i}
            for W in {rule.antecedent for rule in rules if rule.is_frequent(sdb, self.minsup)}
        }

    def _find_right_equivalence_classes(self,
                                        i: int,
                                        rules: Set[Rule],
                                        sdb: Sequence[Sequence]) -> Dict[Itemset, Set[Rule]]:
        return {
            W: {rule for rule in rules if rule.consequent == W and len(rule.antecedent) == i}
            for W in {rule.consequent for rule in rules if rule.is_frequent(sdb, self.minsup)}
        }

    def _first_scan(self,
                    sdb: Sequence[Sequence],
                    ) -> Tuple[Dict[Itemset, Set[Rule]], Dict[Itemset, Set[Rule]]]:
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

    def _left_search(self, leq: Set[Rule], sdb: Sequence[Sequence]) -> None:
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

    def _right_search(self, req: Set[Rule], sdb: Sequence[Sequence]) -> None:
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

    def fit(self, sdb: Sequence[Sequence]) -> None:
        leq, req = self._first_scan(sdb)
        if not self.single_consequent:
            for H in tqdm(leq.values()):
                self._left_search(H, sdb)
        for J in tqdm(req.values()):
            self._right_search(J, sdb)
        if not self.single_consequent:
            for K in tqdm(self._left_store.values()):
                self._left_search(K, sdb)

    def rules_to_df(self, csv_file: str) -> None:
        df = pd.DataFrame(
            [
                [list(r.antecedent), list(r.consequent), r.support, r.confidence]
                for r in self.valid_rules
            ],
            columns=['antecedent', 'consequent', 'support', 'confidence']
        )
        df = df.sort_values(by=['confidence', 'support'], ascending=False)
        df.to_csv(csv_file, index=False)
