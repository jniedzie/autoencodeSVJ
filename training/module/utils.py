import os, random
import numpy as np
import tensorflow as tf
import importlib, sys


def set_random_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def import_class(class_path):
    class_path = class_path.replace(".py", "").replace("/", ".")
    model_module = importlib.import_module(class_path)
    
    model_class = None
    
    model_class_name = class_path.split(".")[-1]
    
    for x in dir(model_module):
        if x == model_class_name:
            model_class = getattr(model_module, x)

    setattr(sys.modules[__name__], model_class.__name__, model_class)
    
    return model_class
