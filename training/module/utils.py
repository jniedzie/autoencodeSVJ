import os
import random
import importlib
import sys

import numpy as np
import tensorflow as tf


def set_random_seed(seed_value):
    """
    
    Sets seed for Python, random library, NumPy and TensorFlow to given value.
    Configures new global TensorFlow session.
    
    Args:
        seed_value (int): Value of the seed
    """
    
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # TODO: Check if this is still needed
    # Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def import_class(class_path):
    """
    Imports and returns class based on path to python file that contains it. Name of the class should be the same
    as name of the file.
    
    Args:
        class_path (str): Path to python file containing a class

    Returns:
        (type): Class type of the loaded class.
    """
    
    class_path = class_path.replace(".py", "").replace("/", ".")
    model_module = importlib.import_module(class_path)
    
    model_class = None
    model_class_name = class_path.split(".")[-1]
    
    for x in dir(model_module):
        if x == model_class_name:
            model_class = getattr(model_module, x)

    setattr(sys.modules[__name__], model_class.__name__, model_class)
    
    return model_class
