"""
Created on 2016-12-22

@author: Peer Springstübe
"""

import os
import hyperopt
import logging
import re
import uuid
from string import Template
import shutil
import subprocess
import hyperopt.pyll.stochastic

tracker_template = "/informatik2/students/home/3springs/git/hiob/hiob/tracker_template.yaml"
tracker_template = "tracker_template.yaml"

# Set up logging:
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


space_vals = {
    #    'conv4_3_cnt': hyperopt.hp.quniform('conv4_3_cnt', 0, 512, 1),
    #    'conv5_3_cnt': hyperopt.hp.quniform('conv5_3_cnt', 0, 512, 1),
    #    'sigma_train': hyperopt.hp.uniform('sigma_train', 0.0, 2.0),
    #    'sigma_update': hyperopt.hp.uniform('sigma_update', 0.0, 2.0),
    #    'update_initial_factor': hyperopt.hp.uniform('update_initial_factor', 0.0, 1.0),
    #    'update_current_factor': hyperopt.hp.uniform('update_current_factor', 0.0, 1.0),
    #    'update_threshold': hyperopt.hp.uniform('update_threshold', 0.0, 1.0),
    #    'update_use_quality': hyperopt.hp.choice('update_use_quality', ['true', 'false']),
    #    'particle_count': hyperopt.hp.quniform('particle_count', 100, 2000, 100),
    #
    'cons_learning_rate': hyperopt.hp.choice('cons_learning_rate', [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]),
    'cons_max_iterations': hyperopt.hp.quniform('cons_max_iterations', 10, 100, 10),
    'cons_min_cost': hyperopt.hp.choice('cons_min_cost', ['null', 0.001, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5]),
    #
    'cons_conv1_channels': hyperopt.hp.choice('cons_conv1_channels', [16, 32, 64, 72]),
    'cons_conv1_kernel_size': hyperopt.hp.choice('cons_conv1_kernel_size', [3, 5, 7, 9, 11, 13, 15]),
    'cons_conv2_kernel_size': hyperopt.hp.choice('cons_conv2_kernel_size', [3, 5, 7, 9, 11, 13, 15]),
}
space_ints = {'conv4_3_cnt', 'conv5_3_cnt', 'particle_count',
              'cons_max_iterations', 'cons_conv1_channels', 'cons_conv1_kernel_size', 'cons_conv2_kernel_size',
              }
space = []
space_keys = []
for k, v in space_vals.items():
    space.append(v)
    space_keys.append(k)


def read_score_from_log(path):
    with open(path, 'r') as f:
        p = re.compile(r".*\bprobe_score=(\d\.\d+)\b.*")
        for line in f.readlines():
            m = p.match(line)
            if m:
                s = m.group(1)
                score = float(s)
                return score
    return None


def create_tracker_conf(path, values):
    with open(tracker_template, 'r') as f:
        tmpl = Template("".join(f.readlines()))
    s = tmpl.substitute(values)
    with open(path, 'w') as f:
        f.write(s)


def get_yaml(args):
    d = {}
    for n, v in enumerate(args):
        key = space_keys[n]
        if key in space_ints:
            d[key] = int(v)
        else:
            if type(v) == float:
                d[key] = "%.10f" % v
            else:
                d[key] = v
    with open(tracker_template, 'r') as f:
        tmpl = Template("".join(f.readlines()))
    s = tmpl.substitute(d)
    return s


def one_track(args):
    conf = get_yaml(args)

    exid = "hyper_hiob_" + str(uuid.uuid4())
    print("Execution: %s" % exid)
    d_path = os.path.join('/', 'tmp', exid)
    t_path = os.path.join(d_path, "tracker.yaml")
    e_path = os.path.join(d_path, "environment.yaml")
    E_path = os.path.join(d_path, "evaluation.log")
    print(d_path, t_path, e_path)
    os.mkdir(d_path)
    # shutil.copyfile(
    #    '/informatik2/students/home/3springs/git/hiob/hiob/tracker.yaml', t_path)
    with open(t_path, "w") as f:
        f.write(conf)
    shutil.copyfile(
        '/informatik2/students/home/3springs/git/hiob/hiob/environment.yaml', e_path)
    subprocess.call(
        ['python', 'hiob_cli.py', '-e', e_path, '-t', t_path, '-E', E_path])
    score = read_score_from_log(E_path)
    print("SCORE:", score)
    return -score


# with open(tracker_template, 'r') as f:
#    tmpl = Template("".join(f.readlines()))
#    print(tmpl.substitute(conv4_3_cnt=23, conv5_3_cnt=17))

# one_track()


best = hyperopt.fmin(
    fn=one_track,
    space=space,
    algo=hyperopt.tpe.suggest,
    max_evals=100,
)
print("BEST:", best)
exit()

for _ in range(10):
    print(hyperopt.pyll.stochastic.sample(space))
