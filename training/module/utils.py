import os, glob, subprocess, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams.update({'font.size': 18})

def save_aucs_to_csv(aucs, path, append=False, write_header=True):
    # TODO: we could simplify what we store in the aucs file to m, r and auc only
    
    with open(path, "a" if append else "w") as out_file:
        if write_header:
            out_file.write(",name,auc,mass,nu\n")
        
        for index, dict in enumerate(aucs):
            out_file.write("{},Zprime_{}GeV_{},{},{},{}\n".format(index,
                                                                  dict["mass"], dict["rinv"], dict["auc"],
                                                                  dict["mass"], dict["rinv"]))


def smartpath(path):
    if path.startswith("~/"):
        return path
    return os.path.abspath(path)


def glob_in_repo(globstring):
    info = {}
    info['head'] = subprocess.Popen("git rev-parse --show-toplevel".split(), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE).communicate()[0].decode("utf-8").strip('\n')
    info['name'] = subprocess.Popen("git config --get remote.origin.url".split(), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE).communicate()[0].decode("utf-8").strip('\n')
    
    repo_head = info['head']
    files = glob.glob(os.path.abspath(globstring))
    
    if len(files) == 0:
        files = glob.glob(os.path.join(repo_head, globstring))
    
    return files


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
