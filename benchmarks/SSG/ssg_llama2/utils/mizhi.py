import os
import sys
import random
import time
from pprint import pprint


class Printer():
    def __init__(self, filename="log.txt", mode='w'):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.terminal = sys.stdout
        self.filename = filename
        with open(self.filename, mode):
            pass

    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, 'a') as f:
            f.write(message)

    def flush(self):
        pass


class Errorer():
    def __init__(self, filename="error.txt", mode='w'):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.terminal = sys.stderr
        self.filename = filename
        with open(self.filename, mode):
            pass

    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, 'a') as f:
            f.write(message)

    def flush(self):
        pass
