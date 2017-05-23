"""
Created on 2016-12-14

@author: Peer Springstübe
"""
from hiob.rect import Rect
import matplotlib

RUN_DIR = "/data/Peer/FCNT-eval/run3"
GT_DIR = "/data/Peer/data/tb100_unzipped"
ENDING = '.position.mat'

import os
import scipy.io
import numpy as np
import re
from matplotlib import pyplot as plt, matplotlib_fname

tb50 = """Ironman
Matrix
MotorRolling
Soccer
Skiing
Freeman4
Freeman1
Skating1
Tiger2
Liquor
Coke
Football
FleetFace
Couple
Tiger1
Woman
Bolt
Freeman3
Basketball
Lemming
Singer2
Subway
CarScale
David3
Shaking
Sylvester
Girl
Jumping
Trellis
David
Boy
Deer
FaceOcc2
Dudek
Football1
Suv
Jogging.1
Jogging.2
MountainBike
Crossing
Singer1
Dog1
Walking
Walking2
Doll
Car4
David2
CarDark
Mhyang
FaceOcc1
Fish"""

tb50_list = tb50.split("\n")
tb50_set = set(tb50_list)


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
    return f


def load_rects(path):
    a = scipy.io.loadmat(path)
    mrects = a['rects']
    r = []
    for i in range(len(mrects[0])):
        mr = mrects[0][i], mrects[1][i], mrects[2][i], mrects[3][i]
        r.append(Rect(mr))
    return r


def load_gt(sname):
    path = os.path.join(GT_DIR, sname, 'groundtruth_rect.txt')
    r = re.compile(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
    rects = []
    with open(path, 'r') as tf:
        for line in tf:
            m = r.match(line)
            if m:
                rect = Rect(tuple(int(m.group(n)) for n in (1, 2, 3, 4)))
            else:
                rect = None
            rects.append(rect)
    return rects

all_rs = []
all_gts = []
tb50_rs = []
tb50_gts = []
tbnew_rs = []
tbnew_gts = []

run_dirs = ["/data/Peer/FCNT-eval/run1",
            "/data/Peer/FCNT-eval/run2", "/data/Peer/FCNT-eval/run3"]
#run_dirs = ["/data/Peer/FCNT-eval/run3", ]
for run_dir in run_dirs:
    for fname in os.listdir(run_dir):
        if not fname.endswith(ENDING):
            continue
        sname = fname[:-len(ENDING)]

        rs_path = os.path.join(run_dir, sname + ENDING)
        rs = load_rects(rs_path)
        gts = load_gt(sname)
        if sname == 'Football1':
            rs = rs[0:74]
        elif sname == 'Freeman4':
            rs = rs[0:283]
        elif sname == 'Freeman3':
            rs = rs[0:460]
        elif sname == 'Diving':
            rs = rs[0:215]
        elif sname == 'David':
            rs = rs[299:]
            gts = gts[299:]
        assert len(rs) == len(gts)
        all_rs.extend(rs)
        all_gts.extend(gts)
        if sname in tb50_set:
            tb50_rs.extend(rs)
            tb50_gts.extend(gts)
        else:
            tbnew_rs.extend(rs)
            tbnew_gts.extend(gts)
        print("Sample:", sname, len(rs), len(gts), len(all_gts))
print(len(all_rs), len(tb50_rs))
# print(all_rs)
# print(all_gts)
cd = np.empty(len(all_rs))
ov = np.empty(len(all_gts))
print(len(cd), len(ov))
for n, (r, gt) in enumerate(zip(all_rs, all_gts)):
    if gt is None:
        print(n)
    cd[n] = r.center_distance(gt)
    ov[n] = r.overlap_score(gt)
dfun = build_dist_fun(cd)
ofun = build_over_fun(ov)

tb50_cd = np.empty(len(tb50_rs))
tb50_ov = np.empty(len(tb50_gts))
print(len(tb50_cd), len(tb50_ov))
for n, (r, gt) in enumerate(zip(tb50_rs, tb50_gts)):
    if gt is None:
        print(n)
    tb50_cd[n] = r.center_distance(gt)
    tb50_ov[n] = r.overlap_score(gt)
tb50_dfun = build_dist_fun(tb50_cd)
tb50_ofun = build_over_fun(tb50_ov)

tbnew_cd = np.empty(len(tbnew_rs))
tbnew_ov = np.empty(len(tbnew_gts))


"""
Created on 2017-04-19

@author: Peer Springstübe
"""

import yaml
import os
from collections import OrderedDict
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, matplotlib_fname

ct = {
    'IV': 'Illumination Variation',
    'SV': 'Scale Variation',
    'OCC': 'Occlusion',
    'DEF': 'Deformation',
    'MB': 'Motion Blur',
    'FM': 'Fast Motion',
    'IPR': 'In-Plane Rotation',
    'OPR': 'Out-of-Plane Rotation',
    'OV': 'Out-of-View',
    'BC': 'Background Clutters',
    'LR': 'Low Resolution',
}


def load_sample_def():
    with open('../conf/collections/tb100_paper50.yaml', 'rt') as f:
        tb50_list = yaml.load(f)['samples']
    with open('../conf/data_sets/tb100.yaml', 'rt') as f:
        tbdef = yaml.load(f)
    samples = OrderedDict()
    attributes = set()
    for sample in tbdef['samples']:
        name = sample['name']
        if 'tb100/' + name in tb50_list:
            sample['50'] = 'p50'
            sample['attributes'].append('p50')
        else:
            sample['50'] = 'n50'
            sample['attributes'].append('n50')
        sample['attributes'].append('full')
        samples[name] = sample
        attributes.update(sample['attributes'])
    return samples, sorted(attributes)


def dir_by_prefix(prefix):
    base = '/data/Peer/hiob_total_recall'
    for d in os.listdir(base):
        if d.startswith(prefix + '-'):
            return os.path.join(base, d)
    return None


def tracking_dirs(base):
    a = []
    for d in os.listdir(base):
        if d.startswith('tracking-'):
            a.append(os.path.join(base, d))
    return sorted(a)


def tracking_log(base):
    tpath = os.path.join(base, 'tracking_log.txt')
    l = []
    with open(tpath, 'rt') as f:
        for line in f.readlines():
            line = line.strip()
            parts = line.split(',')
            l.append({
                'n': int(parts[0]),
                'confidence': float(parts[5]),
                'distance': float(parts[10]),
                'overlap': float(parts[11]),
                'lost': int(parts[12]),
                'update': parts[13],
            })
    return l


def array_from_bag(b):
    dists = np.zeros(len(b), dtype='float32')
    overs = np.zeros(len(b), dtype='float32')
    for n, e in enumerate(b):
        dists[n] = e['distance']
        overs[n] = e['overlap']
    return dists, overs

size = (6, 5)
#size = (12, 10)
#size = (10 * .8, 8 * .8)
DIST = False
run_names = ['t33', 't15', ]
colours = ['blue', 'orange', 'green', 'blue', 'cyan', 'magenta', 'green']
if DIST:
    title = 'Precision of HIOB and FCNT'
else:
    title = 'Success of HIOB and FCNT'
run_long = {
    't17': 'no update',
    't24': 'continuous',
    't18': 'static',
    't09': 'dynamic',
    't15': 'combined',
    't28': 'LD1, lim. dyn. 0.1-0.4',
    't25': 'LD2, lim. dyn. 0.2-0.4',
    't30': 'LC1, lim. comb. 0.1-0.4',
    't33': 'LC2, lim. comb. 0.2-0.4',
}
run_long = {
    't17': 'no update',
    't24': 'continuous',
    't18': 'static',
    't09': 'dynamic',
    't15': 'combined',
    't28': 'LD1',
    't25': 'LD2',
    't30': 'LC1',
    't33': 'limited combined',
    'fcnt': 'FCNT',
}
xrun_long = {
    't17': 'None',
    't24': 'Cont',
    't18': 'Stat',
    't09': 'Dyn',
    't15': 'Cmb',
    't28': 'LD1',
    't25': 'LD2',
    't30': 'LC1',
    't33': 'LC2',
}

run_bags = OrderedDict()
samples, attributes = load_sample_def()
csamples = None
for run in run_names:
    bags = {name: [] for name in attributes}
    csamples = {name: 0 for name in attributes}
    ts = tracking_dirs(dir_by_prefix(run))
    for t in ts:
        tname = t.split('-')[-1]
        sample = samples[tname]
        tlog = tracking_log(t)
        print(t, sample['attributes'])
        for nam in sample['attributes']:
            bags[nam].extend(tlog)
            csamples[nam] += 1
        run_bags[run] = bags


def dist_fig(title=None, size=None,):
    if size is None:
        size = (6, 5)
    f = plt.figure(figsize=size)
    if title is not None:
        plt.title(title)
    plt.xlabel("center distance [pixels]")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=50)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.axvline(x=20, linestyle=':', color='black')


def dist_x():
    return np.arange(0., 50.1, .1)


def dist_plot(dists, col):
    dfun = build_dist_fun(dists)
    at20 = dfun(20)
    x = dist_x()
    y = [dfun(a) for a in x]
    plt.plot(x, y, col)
    return at20


def over_x():
    return np.arange(0., 1.001, 0.001)


def over_fig(title=None, size=None,):
    if size is None:
        size = (6, 5)
    f = plt.figure(figsize=size)
    if title is not None:
        plt.title(title)
    plt.xlabel("overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)


def over_plot(overs, col):
    ofun = build_over_fun(overs)
    x = over_x()
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    plt.plot(x, y, col)
    return auc
# --


def single_dist_plot(data, title, legend=None, size=None, colour='blue'):
    dist_fig(title, size)
    at20 = dist_plot(data, colour)
    if legend is None:
        ltext = 'prec(20) = %0.3f' % at20
    else:
        ltext = legend + ' - prec(20) = %0.3f' % at20
    all_h = matplotlib.patches.Patch(
        color='blue', label=ltext)
    plt.legend(handles=[all_h, ], loc='lower right')
    return at20


def single_compare_dist_plot(data, full, title, size=None):
    dist_fig(title, size)
    at20f = dist_plot(full, 'blue')
    at20 = dist_plot(data, 'red')
    data_h = matplotlib.patches.Patch(
        color='red', label='      prec(20) = %0.3f' % at20)
    all_h = matplotlib.patches.Patch(
        color='blue', label='full prec(20) = %0.3f' % at20f)
    plt.legend(handles=[data_h, all_h, ], loc='lower right')
    return at20


def tb100_dist_plot(p50, n50, title, size=None):
    dist_fig(title, size)
    pat20 = dist_plot(p50, 'r-')
    fat20 = dist_plot(np.append(p50, n50), 'b-')
    nat20 = dist_plot(n50, 'g-')
    tb50_h = matplotlib.patches.Patch(
        color='red', label='tb100(p50) - prec(20) = %0.3f' % pat20)
    all_h = matplotlib.patches.Patch(
        color='blue', label='tb100(full)  - prec(20) = %0.3f' % fat20)
    tbnew_h = matplotlib.patches.Patch(
        color='green', label='tb100(n50) - prec(20) = %0.3f' % nat20)
    plt.legend(handles=[tb50_h, all_h, tbnew_h], loc='lower right')


def tb100_over_plot(p50, n50, title, size=None):
    over_fig(title, size)
    pauc = over_plot(p50, 'r-')
    fauc = over_plot(np.append(p50, n50), 'b-')
    nauc = over_plot(n50, 'g-')
    tb50_h = matplotlib.patches.Patch(
        color='red', label='tb100(p50) - AUC = %0.3f' % pauc)
    all_h = matplotlib.patches.Patch(
        color='blue', label='tb100(full)  - AUC = %0.3f' % fauc)
    new_h = matplotlib.patches.Patch(
        color='green', label='tb100(n50) - AUC = %0.3f' % nauc)
    plt.legend(handles=[tb50_h, all_h, new_h], loc='lower left')


def single_over_plot(data, title, legend=None, size=None, colour='blue'):
    over_fig(title, size)
    auc = over_plot(data, colour)
    if legend is None:
        ltext = 'AUC = %0.3f' % auc
    else:
        ltext = legend + ' - AUC = %0.3f' % auc
    all_h = matplotlib.patches.Patch(
        color='blue', label=ltext)
    plt.legend(handles=[all_h, ], loc='lower left')
    return auc


def single_compare_over_plot(data, full, title, size=None):
    over_fig(title, size)
    aucf = over_plot(full, 'blue')
    auc = over_plot(data, 'red')
    data_h = matplotlib.patches.Patch(
        color='red', label='      AUC = %0.3f' % auc)
    all_h = matplotlib.patches.Patch(
        color='blue', label='full AUC = %0.3f' % aucf)
    plt.legend(handles=[data_h, all_h, ], loc='lower left')
    return auc


def cnt_updates(bag):
    c = 0
    f = 0
    for e in bag:
        u = e['update']
        if u == 'c':
            c += 1
        elif u == 'f':
            f += 1
    return c, f

#size = (5, 4.6)
lines = ['#attribute,precision,overlap,samples,frames,cups,fups\n']
tab = []
outpath = '/data/Peer/hiob_ev'

#tb100_dist_plot(pdists, ndists, 'Precision plot - %s' % (title), size=size)


aucs = {}
at20s = {}

if DIST:
    dist_fig(title, size)
    fcnt_at20 = dist_plot(cd, col='green')
else:
    over_fig(title, size)
    fcnt_auc = over_plot(ov, col='green')

n = 0
for run, bags in run_bags.items():
    colour = colours[n]
    n += 1
    dists, overs = array_from_bag(bags['full'])
    if DIST:
        at20 = dist_plot(dists, col=colour)
        at20s[run] = [at20, colour]
    else:
        auc = over_plot(overs, col=colour)
        aucs[run] = [auc, colour]

if DIST:
    hs = []
    for run in run_names:
        at20, colour = at20s[run]
        name = run_long[run]
        h = matplotlib.patches.Patch(
            color=colour, label='%0.3f - %s' % (at20, name))
        hs.append(h)
    h = matplotlib.patches.Patch(
        color='red', label='%0.3f - %s' % (fcnt_at20, 'FCNT'))
    hs.append(h)
    plt.legend(handles=hs, loc='lower right')
    plt.savefig(os.path.join(outpath, "updates_precision.pdf"))
else:
    hs = []
    for run in run_names:
        auc, colour = aucs[run]
        name = run_long[run]
        h = matplotlib.patches.Patch(
            color=colour, label='%0.3f - %s' % (auc, name))
        hs.append(h)
    h = matplotlib.patches.Patch(
        color='red', label='%0.3f - %s' % (fcnt_auc, 'FCNT'))
    hs.append(h)
    plt.legend(handles=hs, loc='lower left')
    plt.savefig(os.path.join(outpath, "updates_success.pdf"))
plt.show()
exit()
