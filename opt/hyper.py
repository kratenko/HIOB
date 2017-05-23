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

# Set up logging:
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

TEMPLATE = None

space = []
space_keys = []


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


def get_yaml_template(args, tmpl):
    global space, space_keys
    d = {}
    for n, v in enumerate(args):
        key = space_keys[n]
        if key.endswith('_int'):
            d[key] = int(v)
        else:
            if type(v) == float:
                d[key] = "%.10f" % v
            else:
                d[key] = v
    s = tmpl.substitute(d)
    return s


def one_track(args):
    global TEMPLATE
    conf = get_yaml_template(args, TEMPLATE)

    exid = "hyper_hiob_" + str(uuid.uuid4())
    print("Execution: %s" % exid)
    d_path = os.path.join('/', 'tmp', exid)
    t_path = os.path.join(d_path, "tracker.yaml")
    e_path = os.path.join(d_path, "environment.yaml")
    E_path = os.path.join(d_path, "evaluation.log")
    print(d_path, t_path, e_path)
    os.mkdir(d_path)
    with open(t_path, "w") as f:
        f.write(conf)
    shutil.copyfile(
        'environment.yaml', e_path)
#        '/informatik2/students/home/3springs/git/hiob/hiob/environment.yaml', e_path)
    subprocess.call(
        ['python', 'hiob_cli.py', '-e', e_path, '-t', t_path, '-E', E_path],
        cwd='../hiob',
    )
    score = read_score_from_log(E_path)
    print("SCORE:", score)
    return -score


def run(space_vals, max_evals, template):
    global space, space_keys, TEMPLATE
    space = []
    space_keys = []
    for k, v in space_vals.items():
        space.append(v)
        space_keys.append(k)
    TEMPLATE = Template(template)
    best = hyperopt.fmin(
        fn=one_track,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
    )
    print("BEST:", best)
