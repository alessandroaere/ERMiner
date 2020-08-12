from core.sequential_pattern_mining import ERMiner
import argparse
import os
from core.config import DEFAULT_PARAMS


parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str)
parser.add_argument('--outpath', type=str, default=os.path.join(os.getcwd(), 'data', 'rules.csv'))
parser.add_argument('--itemsep', type=str, default=' -1 ')
parser.add_argument('--sequencesep', type=str, default=' -1 -2\n')
args = parser.parse_args()

if __name__ == '__main__':

    with open(args.datapath, 'r') as f:
        data = f.read()
    data = data.split(args.sequencesep)[:10]
    data = [s.split(args.itemsep) for s in data]

    model = ERMiner(minsup=DEFAULT_PARAMS['minsup'],
                    minconf=DEFAULT_PARAMS['minconf'],
                    single_consequent=True)

    model.fit(data)
    model.rules_to_df(args.outpath)
