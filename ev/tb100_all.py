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


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
    return f


def array_from_bag(b):
    dists = np.zeros(len(b), dtype='float32')
    overs = np.zeros(len(b), dtype='float32')
    for n, e in enumerate(b):
        dists[n] = e['distance']
        overs[n] = e['overlap']
    return dists, overs

#size = (6, 5)
#size = (12, 10)
size = (10 * .8, 8 * .8)
DIST = True
run_names = ['t17', 't24', 't18', 't09', 't15', 't25', 't33']
colours = ['cyan', 'magenta', 'green', 'blue', 'red', 'orange', 'black']
run_names = ['t33', 't25', 't15', 't09', 't18', 't24', 't17']
colours = ['black', 'orange', 'red', 'blue', 'cyan', 'magenta', 'green']
if DIST:
    title = 'Precision of different update strategies'
else:
    title = 'Success of different update strategies'
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
    't33': 'LC2',
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


if DIST:
    dist_fig(title, size)
else:
    over_fig(title, size)

n = 0
aucs = {}
at20s = {}
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
    plt.legend(handles=hs, loc='lower left')
    plt.savefig(os.path.join(outpath, "updates_success.pdf"))
plt.show()
exit()


plt.savefig(os.path.join(outpath, "precision.pdf"))
plt.close()
single_over_plot(fovers, "Success plot - Example", size=size,)
plt.savefig(os.path.join(outpath, "success.pdf"))
plt.close()
exit()


tb100_dist_plot(pdists, ndists, 'Precision plot - %s' % (title), size=size)
plt.savefig(os.path.join(outpath, "precision.pdf"))
plt.close()
tb100_over_plot(povers, novers, 'Success plot - %s' % (title), size=size)
plt.savefig(os.path.join(outpath, "success.pdf"))
plt.close()

for nam in sorted(bags):
    bag = bags[nam]
    dists, overs = array_from_bag(bag)
    at20 = single_compare_dist_plot(
        dists, fdists, 'Precision plot - %s - %s ' % (title, nam), size=size)
    plt.savefig(os.path.join(outpath, "ATT-%s-precision.pdf" % nam))
    plt.close()
#    auc = single_over_plot(overs, nam, nam)
    auc = single_compare_over_plot(
        overs, fovers, 'Success plot - %s - %s ' % (title, nam), size=size)
    plt.savefig(os.path.join(outpath, "ATT-%s-success.pdf" % nam))
    plt.close()
    cs, fs = cnt_updates(bag)
    lines.append('%s,%.3f,%.3f,%d,%d,%d,%d\n' %
                 (nam, at20, auc, csamples[nam], len(bag), cs, fs))
    tab.append("\t\t%-12s & %3d & %5d & %.3f & %.3f \\\\\n" %
               (nam, csamples[nam], len(bag), at20, auc))
print("".join(lines))
with open(os.path.join(outpath, 'eval.csv'), 'wt') as f:
    f.writelines(lines)
with open(os.path.join(outpath, 'tab.txt'), 'wt') as f:
    f.writelines(tab)
exit()

dist_fig(title='New one distance')
at20 = dist_plot(dists, 'b-')
all_h = matplotlib.patches.Patch(
    color='blue', label='tb100(p50)  - prec(20) = %0.3f' % at20)
plt.legend(handles=[all_h, ], loc='lower right')
# plt.savefig("precision_plot.pdf")
# plt.savefig("precision_plot.svg")
# plt.show()

over_fig(title='New one overlap')
auc = over_plot(overs, 'b-')
all_h = matplotlib.patches.Patch(
    color='blue', label='tb100(p50)  - AUC = %0.3f' % auc)
plt.legend(handles=[all_h, ], loc='lower left')
# plt.savefig("success_plot.pdf")
# plt.savefig("success_plot.svg")
plt.show()
