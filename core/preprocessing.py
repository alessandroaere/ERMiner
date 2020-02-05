import pandas as pd
import itertools


class History:

    @staticmethod
    def is_sorted(mylist):
        return all(mylist[i] <= mylist[i + 1] for i in range(len(mylist) - 1))

    @staticmethod
    def read_csv(csv_file, *args, **kwargs):
        return pd.read_csv(csv_file, *args, **kwargs)

    @staticmethod
    def get_event_sequence(data, event_column):
        return data[event_column].tolist()

    @staticmethod
    def get_timestamp(data, time_columns):
        if len(time_columns) == 2:
            return [t for t in (data[time_columns[0]] * 60 + data[time_columns[1]]).tolist()]
        elif len(time_columns) == 1:
            return [int(t/60) for t in data[time_columns[0]].tolist()]
        else:
            raise ValueError("Invalid argument value: time_columns must be a 1-sized or 2-sized list")

    @staticmethod
    def get_context(data, contexts):
        return [tuple(x) for x in data[contexts].values]

    @staticmethod
    def reshape(obj, like):
        out = []
        i = 0
        for l in like:
            out.append(obj[i:i+len(l)])
            i += len(l)
        return out

    @staticmethod
    def filter_single_event(obj):
        return [s for s in obj if len(s) > 1]

    def __init__(self, csv_file, event_column, time_columns, contexts, *args, **kwargs):
        self.data = self.read_csv(csv_file, *args, **kwargs) if isinstance(csv_file, str) else csv_file
        self.event_sequences = [self.get_event_sequence(self.data, event_column)]
        self.timestamp = self.get_timestamp(self.data, time_columns)
        self.current_sequence = None
        self.contexts = self.get_context(self.data, contexts)
        self.contexts_name = contexts

    def split_by_variable(self, variable):

        self.event_sequences = [[event for _, event in seq] for sequence in self.event_sequences
                                for _, seq in itertools.groupby(zip(self.data[variable], sequence),
                                                                lambda x: x[0])]
        self.current_sequence = self.event_sequences[-1]
        return self.event_sequences

    def split_by_event(self, event, position):

        output_sequences = []

        for sequence in self.event_sequences:
            indices = [i for i, x in enumerate(sequence) if x == event]
            if position == 'first':
                output_sequences += [sequence[i: j] for i, j in zip([0] + indices, indices + [None])][1:]
            elif position == 'last':
                indices = [i + 1 for i in indices]
                output_sequences += [sequence[i: j] for i, j in zip([0] + indices, indices + [None])][:-1]
            else:
                raise ValueError("Invalid argument value: position must be 'first' or 'last'")

        self.event_sequences = output_sequences
        self.timestamp = self.reshape(obj=self.timestamp, like=self.event_sequences)
        self.contexts = self.reshape(obj=self.contexts, like=self.event_sequences)
        self.current_sequence = self.event_sequences[-1] if self.event_sequences else []
        return self.event_sequences

    def split_by_timestamp(self, window_size, slicing, slice_on_time=True, drop_monosequences=True):

        output_sequences = []
        output_timestamps = []
        output_contexts = []
        for sequence, time, contexts in zip(self.event_sequences, self.timestamp, self.contexts):

            if not self.is_sorted(time):
                raise ValueError('Invalid timestamp values: timestamp must be sorted {}'.format(time))

            pointer = 0
            while pointer < len(sequence) - 1:
                last_t = len([t for t in time if t <= time[pointer] + window_size])
                output_sequences.append(sequence[pointer: last_t])
                output_timestamps.append(time[pointer: last_t])
                output_contexts.append(contexts[pointer: last_t])
                if slice_on_time:
                    pointer = len([t for t in time if t < time[pointer] + slicing])
                else:
                    pointer = pointer + slicing

        if drop_monosequences:
            self.event_sequences = self.filter_single_event(output_sequences)
            self.timestamp = self.filter_single_event(output_timestamps)
            self.contexts = self.filter_single_event(output_contexts)
        else:
            self.event_sequences = output_sequences
            self.timestamp = output_timestamps
            self.contexts = output_contexts
        return self.event_sequences

    def get_current_window(self, current_time_sequence, window_size):

        index = 0
        while current_time_sequence[-1] - current_time_sequence[index] > window_size:
            index += 1

        return self.current_sequence[index:]
