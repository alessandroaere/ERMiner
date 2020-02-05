from .sequential_pattern_mining import ERMiner, RulesDatabase
from .preprocessing import History
import os
import json
import pandas as pd
from .config import DEFAULT_PARAMS


class HMIApp:

    @staticmethod
    def row_to_df(row):
        dict_file = json.loads(row.to_json())
        return pd.DataFrame({'actionid': [dict_file['actionid']],
                             'timestamp': [dict_file['timestamp']],
                             'day': [dict_file['day']],
                             'hours': [dict_file['hours']]})

    @staticmethod
    def payload_to_df(payload):
        actionid_list = [action['actionid'] for action in payload['sequence']]
        timestamp_list = [action['timestamp'] for action in payload['sequence']]
        day_list = [action['day'] for action in payload['sequence']]
        hours_list = [action['hours'] for action in payload['sequence']]
        return pd.DataFrame({'actionid': actionid_list,
                             'timestamp': timestamp_list,
                             'day': day_list,
                             'hours': hours_list})

    @staticmethod
    def read_csv(csv_file, *args, **kwargs):
        return pd.read_csv(csv_file, *args, **kwargs)

    @staticmethod
    def read_json(json_file, *args, **kwargs):
        with open(json_file) as f:
            data = json.load(f, *args, **kwargs)
        return data

    def __init__(self, user_name, datadir, privacy_level, default_history=None):
        self.user = User(user_name, datadir, default_history)
        self.privacy_level = privacy_level
        self.params = self.read_json(os.path.join(datadir, self.user.name, 'params.json'))
        self.sessions_dict = self.read_json(os.path.join(datadir, 'sessions_dict.json'))
        self.actions_df = self.read_csv(os.path.join(datadir, 'actions_df.csv'), index_col=0)
        self.passive_actions = self.actions_df.loc[self.actions_df['type'] == 'passive']['actionid'].tolist()
        self.history_df = self.read_csv(os.path.join(datadir, user_name, 'history.csv'), index_col=0)
        self.model = ERMiner(minsup=self.params['minsup'],
                             minconf=self.params['minconf'][privacy_level-1],
                             single_consequent=True)
        self.buffer = None
        self.history = None
        self.session_timestamp = None
        self.process_history()

    def process_history(self):
        self.history = History(csv_file=self.history_df,
                               event_column='actionid',
                               time_columns=['timestamp'],
                               contexts=['day', 'hours'])
        self.history.split_by_event(event='951210be4effb39472943498c7bef523', position='first')
        self.session_timestamp = self.history.timestamp
        self.history.split_by_timestamp(window_size=self.params['window_size'], slicing=self.params['slicing'])

    def add_actions_to_history(self, input_data, only_memory=False):
        if isinstance(input_data, dict):
            self.buffer = self.payload_to_df(input_data)
        elif isinstance(input_data, pd.DataFrame):
            self.buffer = input_data
        else:
            raise ValueError("Invalid type for input data. Must be a json or pandas DataFrame.")
        self.history_df = self.history_df.append(self.buffer)
        if not only_memory:
            self.history_df.to_csv(os.path.join(self.user.datadir, self.user.name, 'history.csv'))

    def generate_rules(self):
        self.model.fit(self.history.event_sequences)
        self.model.rules_to_df(os.path.join(self.user.datadir, self.user.name, 'rules.csv'))

    def load_rules(self):
        rules_dir = os.path.join(self.user.datadir, self.user.name, 'rules.csv')
        if os.path.isfile(rules_dir):
            self.model.rules_from_df(rules_dir)
        else:
            self.model.rules = RulesDatabase(set())

    def predict_next_action(self, min_threshold=None, current_day=None, current_hour=None):
        if self.session_timestamp:
            current_window = self.history.get_current_window(current_time_sequence=self.session_timestamp[-1],
                                                             window_size=self.params['window_size'])
            if not current_day or not current_hour:
                if isinstance(self.buffer, pd.DataFrame):
                    current_day = self.buffer['day'].tolist()[-1]
                    current_hour = self.buffer['hours'].tolist()[-1]
                else:
                    raise IOError("Input error: 'current_day' and 'current_hour' must be specified")
            prediction = self.model.predict(sequence=current_window,
                                            hidden_items=[current_window[-1],
                                                          self.sessions_dict[current_window[-1]],
                                                          *self.passive_actions],
                                            c1=self.params['c1'],
                                            c2=self.params['c2'],
                                            c3=self.params['c3'],
                                            c4=self.params['c4'],
                                            current_day=current_day,
                                            current_hour=current_hour,
                                            min_threshold=min_threshold)
            return prediction
        return {'action': None, 'delta': None}

    def simulate_usage(self, type, privacy_level=None, bayesian_optimizer=None, maxevals=10):
        if privacy_level:
            self.privacy_level = privacy_level
            self.model = ERMiner(minsup=self.params['minsup'],
                                 minconf=self.params['minconf'][self.privacy_level - 1],
                                 single_consequent=True)
        self.history_df['json'] = pd.Series([]) if self.history_df.empty \
            else self.history_df.apply(lambda x: self.row_to_df(x), axis=1)
        if type == 2:
            self.generate_rules()
        output = pd.DataFrame(columns=['prediction', 'delta_prediction'])
        list_payloads = self.history_df['json'].tolist()
        self.history_df = pd.DataFrame(columns=['actionid', 'timestamp', 'day', 'hours'])
        for payload in list_payloads:
            self.add_actions_to_history(payload, only_memory=True)
            self.process_history()
            if self.history_df.actionid.iloc[-1] == '951210be4effb39472943498c7bef523' and bayesian_optimizer:
                bayesian_optimizer.monitor.history_df = self.history_df.reset_index(drop=True)
                bayesian_optimizer.optimization_step(maxevals)
                self.params = bayesian_optimizer.best
            if (self.history_df.actionid.iloc[-1] == '951210be4effb39472943498c7bef523' and type == 1) or type == 0:
                self.generate_rules()
            else:
                self.load_rules()
            prediction = self.predict_next_action()
            result = pd.DataFrame({'prediction': [prediction['action']], 'delta_prediction': [prediction['delta']]})
            output = output.append(result)
        output = output.reset_index(drop=True)
        output.to_csv(os.path.join(self.user.datadir, self.user.name, 'result_{}.csv'.format(self.privacy_level)))


class User:

    @staticmethod
    def read_csv(csv_file, *args, **kwargs):
        return pd.read_csv(csv_file, *args, **kwargs)

    @staticmethod
    def read_json(json_file, *args, **kwargs):
        with open(json_file) as f:
            data = json.load(f, *args, **kwargs)
        return data

    def __init__(self, name, datadir, default_history):
        self.name = name
        self.datadir = datadir
        self.default_history = default_history
        self.check()

    def check(self):
        if not os.path.isdir(os.path.join(self.datadir, self.name)):
            self.enroll()
        elif not os.path.exists(os.path.join(self.datadir, self.name, 'history.csv')):
            self.create_history()
        elif not os.path.exists(os.path.join(self.datadir, self.name, 'params.json')):
            self.initialize_params()
        else:
            pass

    def enroll(self):
        os.mkdir(os.path.join(self.datadir, self.name))
        self.create_history()
        self.initialize_params()

    def create_history(self):
        if self.default_history:
            history_df = self.read_csv(os.path.join(self.datadir, self.default_history, 'history.csv'), index_col=0)
        else:
            history_df = pd.DataFrame(columns=['actionid', 'timestamp', 'day', 'hours'])
        history_df.to_csv(os.path.join(self.datadir, self.name, 'history.csv'))

    def initialize_params(self):
        if self.default_history:
            with open(os.path.join(self.datadir, self.name, 'params.json'), 'w') as f:
                params = self.read_json(os.path.join(self.datadir, self.default_history, 'params.json'))
                json.dump(params, f)
        else:
            with open(os.path.join(self.datadir, self.name, 'params.json'), 'w') as f:
                json.dump(DEFAULT_PARAMS, f)
