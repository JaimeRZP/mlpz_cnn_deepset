import sklearn
from packaging import version

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")

from . import data, io, models, qp_output, stats
from .data import *
from .io import *
from .models import *
from .qp_output import *
from .stats import *

__version__ = "0.1.0"
