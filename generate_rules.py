from core.app import HMIApp
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('username', type=str)
parser.add_argument('privacy', type=int)
parser.add_argument('--datadir', type=str, default=os.path.join(os.getcwd(), 'data'))
args = parser.parse_args()

if __name__ == '__main__':

    app = HMIApp(args.username, args.datadir, args.privacy)
    app.generate_rules()
