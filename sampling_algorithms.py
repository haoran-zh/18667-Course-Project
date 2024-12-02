import torch
import numpy as np
import copy

from enum import Enum


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    RELEASE = 3
