from .app import HMIApp
from .preprocessing import History
import pandas as pd
import os
from copy import copy


class Monitor(HMIApp):

    def __init__(self, user_name, datadir, privacy_level, default_history=None):
        super().__init__(user_name, datadir, privacy_level, default_history)
        self.user_name = user_name
        self.datadir = datadir
        self.result_df = None
        self.history_temp = None
        self.history_df = self.history_df.reset_index(drop=True)
        self.history_in_sessions = self.custom_process_history(999)
        self.history_in_windows = self.custom_process_history(self.params['window_size'])
        self.statistics = dict()

    def custom_process_history(self, window_size):
        history = History(csv_file=self.history_df,
                          event_column='actionid',
                          time_columns=['timestamp'],
                          contexts=['day', 'hours'])
        history.split_by_event(event='951210be4effb39472943498c7bef523', position='first')
        history.split_by_timestamp(window_size=window_size,
                                   slicing=1,
                                   slice_on_time=False,
                                   drop_monosequences=False)
        return history

    def print_distinct_actions(self):
        print('Distinct actions in the history: {}'.format(len(self.history_df['actionid'].unique())))

    def check_statistics(self):
        empty_dictionary = {
            'percentage_of_suggestions': 0,
            'next_action_performance': 0,
            'following_window_performance': 0,
            'following_session_performance': 0,
            'delta_error': self.params['window_size'],
            'delta_error_skew': self.params['window_size']
        }
        empty_dictionary.update(self.statistics)
        self.statistics = copy(empty_dictionary)

    def short_history(self, min_rows):
        if self.history_temp.shape[0] < min_rows:
            self.check_statistics()
            return True
        else:
            return False

    def compute_statistics(self):
        self.statistics = dict()
        self.history_temp = self.history_df.reset_index(drop=True)
        self.history_temp['timestamp'] = self.history_temp['timestamp'] // 60
        self.result_df = pd.read_csv(os.path.join(self.datadir,
                                                  self.user_name,
                                                  'result_{}.csv'.format(self.privacy_level)), index_col=0)
        if self.short_history(5):
            return None
        self.history_temp['next_action'] = self.history_temp['actionid'].tolist()[1:] + [None]
        self.history_temp = self.history_temp.reset_index(drop=True)
        self.result_df = self.result_df.reset_index(drop=True)
        self.history_temp = pd.concat([self.history_temp, self.result_df], axis=1)
        self.history_temp['is_prediction'] = self.history_temp['prediction'].notnull()
        self.history_temp = self.history_temp[self.history_temp.actionid.notnull()]
        self.history_temp = self.history_temp[self.history_temp.next_action.notnull()]
        self.history_temp = self.history_temp[self.history_temp.next_action != '951210be4effb39472943498c7bef523']
        self.history_temp['following_window'] = self.history_in_windows.event_sequences
        self.history_temp['following_session'] = self.history_in_sessions.event_sequences
        self.history_temp['next_timestamp'] = [t[1] for t in self.history_in_sessions.timestamp]
        self.history_temp = self.history_temp[~self.history_temp.next_action.isin(self.passive_actions)]
        if self.short_history(3):
            return None
        self.statistics['percentage_of_suggestions'] = self.history_temp['is_prediction'].mean()
        self.history_temp['match'] = self.history_temp.apply(lambda x: x['next_action'] == x['prediction'], axis=1)
        self.history_temp['prediction_in_window'] = self.history_temp.apply(lambda x: x['prediction']
                                                                            in x['following_window'], axis=1)
        self.history_temp['prediction_in_session'] = self.history_temp.apply(lambda x: x['prediction']
                                                                             in x['following_session'],
                                                                             axis=1)
        self.history_temp = self.history_temp[self.history_temp['is_prediction']]
        if self.short_history(3):
            return None
        self.statistics['next_action_performance'] = self.history_temp['match'].mean()
        self.statistics['following_window_performance'] = self.history_temp['prediction_in_window'].mean()
        self.statistics['following_session_performance'] = self.history_temp['prediction_in_session'].mean()
        self.history_temp = self.history_temp[self.history_temp['match']]
        if self.short_history(3):
            return None
        self.history_temp['delta_true'] = self.history_temp.apply(
            lambda x: x['next_timestamp'] - x['timestamp'],
            axis=1)
        self.history_temp['delta_error'] = self.history_temp.apply(
            lambda x: abs(x['delta_prediction'] - x['delta_true']),
            axis=1)
        self.history_temp['delta_error_skew'] = self.history_temp.apply(
            lambda x: 0.5 * abs(x['delta_prediction'] - x['delta_true'])
            if x['delta_prediction'] < x['delta_true'] else 1.5 * abs(x['delta_prediction'] - x['delta_true']), axis=1)
        self.statistics['delta_error'] = self.history_temp['delta_error'].mean()
        self.statistics['delta_error_skew'] = self.history_temp['delta_error_skew'].mean()

    def print_statistics(self):
        for k, v in self.statistics.items():
            print(k, ': ', v)


def monitor_performance(HMIApp):

    class Monitor:

        def __init__(self, *args, **kwargs):
            self.hmi_app = HMIApp(*args, **kwargs)
            self.hmi_app.history_df = self.hmi_app.history_df.reset_index(drop=True)

    return Monitor