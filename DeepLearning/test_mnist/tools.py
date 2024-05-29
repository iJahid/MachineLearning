import time
import numpy as np
from datetime import datetime

class SynchronousTimer:
    def __init__(self):
        self.start_time = time.time()
        self.actions = []

    def add(self, interval_str, action, description=None):
        interval_sec = self._convert_to_seconds(interval_str)

        if (interval_sec < 0):
            return

        self.actions.append({
            "interval": interval_sec,
            "action": action,
            "last_execution": self.start_time,
            "description": description
        })

    def tick(self):
        current_time = time.time()
        for action_info in self.actions:
            if current_time - action_info["last_execution"] >= action_info["interval"]:
                action_info["action"]()  # Invoke the lambda or callable
                action_info["last_execution"] = current_time
                if action_info["description"] is not None:
                    current_time = datetime.now()
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{formatted_time}] {action_info['description']}")

    def _convert_to_seconds(self, time_str):
        number = int(time_str[:-1])  # Extracts all characters except the last
        unit = time_str[-1]  # Takes the last character of the string

        if unit == 's':
            return number
        elif unit == 'm':
            return number * 60
        elif unit == 'h':
            return number * 3600
        else:
            return -1


# Shuffle two arrays maintaining the relative ordering of elements
# Parameters:
#
# Input:    X:  Array of elements to be shuffled randomly
# Input     Y:  Array of elements to be shuffled in the same order as X
# Output        Shuffled X and Y, where the relative ordering of elements is maintained
def shuffle(X, Y):
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]


# Partition an array into batches
# Parameters:
#
# Input:    data:       Array of data to be partitioned into batches
# Input     batch_size: Size of the batches
# Output                Array of batches, containing 'batch_size' elements each
def partition(data, batch_size):
    n = len(data)
    batches = []
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches


# Shuffle the training examples and make new batches of size m
#
# Parameters:
# Input     X_examples:     Array containing all training examples
# Input     Y_examples:     Array containing all training labels
# Input     m:              Batch size
# Output    X_batches:      Batch of m training examples
# Output    Y_batches:      Batch of m corresponding labels
# Output    num_batches:    Number of batches
def make_batches(X_examples, Y_examples, m):
    X_shuffled, Y_shuffled = shuffle(X_examples, Y_examples)
    X_batches = partition(X_shuffled, m)
    Y_batches = partition(Y_shuffled, m)
    num_batches = len(X_batches)

    return X_batches, Y_batches, num_batches