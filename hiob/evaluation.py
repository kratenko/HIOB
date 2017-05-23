"""
Created on 2016-11-30

@author: Peer Springstübe
"""

import os
import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

logger = logging.getLogger(__name__)

#from matplotlib import rcParams
#rcParams['font.family'] = 'serif'
#rcParams['font.size'] = 10


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
    return f


def do_tracking_evaluation(tracking):
    tracker = tracking.tracker

    evaluation = OrderedDict()
    evaluation['set_name'] = tracking.sample.set_name
    evaluation['sample_name'] = tracking.sample.name
    evaluation['loaded'] = tracking.ts_loaded
    evaluation['features_selected'] = tracking.ts_features_selected
    evaluation['consolidator_trained'] = tracking.ts_consolidator_trained
    evaluation['tracking_completed'] = tracking.ts_tracking_completed
    evaluation['total_seconds'] = (
        tracking.ts_tracking_completed - tracking.ts_loaded).total_seconds()
    evaluation['preparing_seconds'] = (
        tracking.ts_consolidator_trained - tracking.ts_loaded).total_seconds()
    evaluation['tracking_seconds'] = (
        tracking.ts_tracking_completed - tracking.ts_consolidator_trained).total_seconds()
    evaluation['sample_frames'] = tracking.total_frames
    evaluation['frame_rate'] = tracking.total_frames / \
        evaluation['total_seconds']

    tracking_dir = os.path.join(tracker.execution_dir, tracking.name)
    try:
        os.makedirs(tracking_dir)
    except:
        logger.error(
            "Could not create tracking log dir '%s', results will be wasted", tracking_dir)

    log = tracking.tracking_log
    princeton_lines = []
    csv_lines = []
    lost1 = 0
    lost2 = 0
    lost3 = 0
    for n, l in enumerate(log):
        r = l['result']
        pos = r['predicted_position']
        roi = l['roi']
        # princeton as they want it:
        if r['lost'] >= 3:
            # lost object:
            line = "NaN,NaN,NaN,NaN"
        else:
            # found, position:
            line = "{},{},{},{}".format(
                pos.left, pos.top, pos.right, pos.bottom)
        princeton_lines.append(line)
        # my own log line:
        line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            n + 1,
            pos.left, pos.top, pos.width, pos.height,
            r['prediction_quality'],
            roi.left, roi.top, roi.width, roi.height,
            r['center_distance'],
            r['overlap_score'],
            r['lost'],
            r['updated']
        )
        csv_lines.append(line)
        if r['lost'] == 1:
            lost1 += 1
        elif r['lost'] == 2:
            lost2 += 1
        elif r['lost'] == 3:
            lost3 += 1
    evaluation['lost1'] = lost1
    evaluation['lost2'] = lost2
    evaluation['lost3'] = lost3
    evaluation['updates_max_frames'] = tracking.updates_max_frames
    evaluation['updates_confidence'] = tracking.updates_confidence
    evaluation['updates_total'] = tracking.updates_max_frames + \
        tracking.updates_confidence

    princeton_filename = os.path.join(
        tracking_dir, tracking.sample.name + '.txt')
    with open(princeton_filename, 'a') as f:
        f.write("\n".join(princeton_lines))
    csv_filename = os.path.join(tracking_dir, "tracking_log" + '.txt')
    with open(csv_filename, 'w') as f:
        f.write("".join(csv_lines))
    dump_filename = os.path.join(tracking_dir, "tracking_log" + '.p')
    with open(dump_filename, 'wb') as f:
        pickle.dump(log, f)

    # figures:
    cd = np.empty(len(log))
    ov = np.empty(len(log))
    cf = np.empty(len(log))
    in20 = 0
    for n, l in enumerate(log):
        r = l['result']
        if (r['center_distance'] is not None) and (r['center_distance'] <= 20):
            in20 += 1
        cd[n] = r['center_distance']
        ov[n] = r['overlap_score']
        cf[n] = r['prediction_quality']

    dim = np.arange(1, len(cd) + 1)

    # distances:
    figure_file2 = os.path.join(tracking_dir, 'center_distance.svg')
    figure_file3 = os.path.join(tracking_dir, 'center_distance.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("center distance")
    plt.axhline(y=20, color='r', linestyle='--')
    plt.plot(dim, cd, 'k', dim, cd, 'bo')
    plt.xlim(1, len(cd))
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)

    figure_file2 = os.path.join(tracking_dir, 'overlap_score.svg')
    figure_file3 = os.path.join(tracking_dir, 'overlap_score.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("overlap score")
    plt.plot(dim, ov, 'k', dim, ov, 'bo')
    plt.xlim(1, len(cd))
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)

    # eval from paper:
    dfun = build_dist_fun(cd)
    ofun = build_over_fun(ov)

    figure_file2 = os.path.join(tracking_dir, 'precision_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'precision_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 50.1, .1)
    y = [dfun(a) for a in x]
    at20 = dfun(20)
    tx = "prec(20) = %0.4f" % at20
    plt.text(5.05, 0.05, tx)
    plt.xlabel("center distance [pixels]")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=50)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    # saving values:
    evaluation['precision_rating'] = at20

    figure_file2 = os.path.join(tracking_dir, 'success_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'success_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    tx = "AUC = %0.4f" % auc
    plt.text(0.05, 0.05, tx)
    plt.xlabel("overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    # saving values:
    evaluation['success_rating'] = auc

    # plot confidence
    figure_file2 = os.path.join(tracking_dir, 'confidence_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'confidence_plot.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("confidence")
    # plt.axhline(y=20, color='r', linestyle='--')
    plt.plot(dim, cf, 'k', dim, cf, 'bo')
    plt.xlim(1, len(cf))
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)

    tracking.evaluation = evaluation
    evaluation_file = os.path.join(tracking_dir, 'evaluation.txt')
    with open(evaluation_file, 'w') as f:
        for k, v in evaluation.items():
            f.write("{}={}\n".format(k, v))


def do_tracker_evaluation(tracker):
    execution_dir = tracker.execution_dir
    trackings_file = os.path.join(execution_dir, 'trackings.txt')
    tracking_sum = 0.0
    preparing_sum = 0.0
    precision_sum = 0.0
    success_sum = 0.0
    lost1 = 0
    lost2 = 0
    lost3 = 0
    updates_max_frames = 0
    updates_confidence = 0
    updates_total = 0
    with open(trackings_file, 'w') as f:
        line = "#n,set_name,sample_name,sample_frames,precision_rating,success_rating,loaded,features_selected,consolidator_trained,tracking_completed,total_seconds,preparing_seconds,tracking_seconds,frame_rate,lost1,lost2,lost3,updates_max_frames,updates_confidence,update_total\n"
        f.write(line)
        for n, e in enumerate(tracker.tracking_evaluations):
            line = "{n},{set_name},{sample_name},{sample_frames},{precision_rating},{success_rating},{loaded},{features_selected},{consolidator_trained},{tracking_completed},{total_seconds},{preparing_seconds},{tracking_seconds},{frame_rate},{lost1},{lost2},{lost3},{updates_max_frames},{updates_confidence},{updates_total}\n".format(
                n=n + 1,
                **e)
            f.write(line)
            preparing_sum += e['preparing_seconds']
            tracking_sum += e['tracking_seconds']
            precision_sum += e['precision_rating']
            success_sum += e['success_rating']
            lost1 += e['lost1']
            lost2 += e['lost2']
            lost3 += e['lost3']
            updates_max_frames += e['updates_max_frames']
            updates_confidence += e['updates_confidence']
            updates_total += e['updates_total']

    # eval from paper:
    dfun = build_dist_fun(tracker.total_center_distances)
    ofun = build_over_fun(tracker.total_overlap_scores)

    figure_file2 = os.path.join(execution_dir, 'precision_plot.svg')
    figure_file3 = os.path.join(execution_dir, 'precision_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 50.1, .1)
    y = [dfun(a) for a in x]
    at20 = dfun(20)
    tx = "prec(20) = %0.4f" % at20
    plt.text(5.05, 0.05, tx)
    plt.xlabel("center distance [pixels]")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=50)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)

    figure_file2 = os.path.join(execution_dir, 'success_plot.svg')
    figure_file3 = os.path.join(execution_dir, 'success_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    tx = "AUC = %0.4f" % auc
    plt.text(0.05, 0.05, tx)
    plt.xlabel("overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)

    ev = OrderedDict()
    ev['execution_name'] = tracker.execution_name
    ev['execution_id'] = tracker.execution_id
    ev['execution_host'] = tracker.execution_host
    ev['execution_dir'] = tracker.execution_dir
    ev['environment_name'] = tracker.environment_name
    ev['git_revision'] = tracker.git_revision
    ev['git_dirty'] = tracker.git_dirty
    ev['random_seed'] = tracker.py_seed
    ev['started'] = tracker.ts_created
    ev['finished'] = tracker.ts_done
    ev['total_samples'] = len(tracker.tracking_evaluations)
    ev['total_frames'] = len(tracker.total_center_distances)
    ev['total_seconds'] = (
        tracker.ts_done - tracker.ts_created).total_seconds()
    ev['average_seconds_per_sample'] = ev[
        'total_seconds'] / ev['total_samples']
    ev['frame_rate'] = ev['total_frames'] / ev['total_seconds']
    ev['preparing_seconds'] = preparing_sum
    ev['tracking_seconds'] = tracking_sum
    apr = precision_sum / ev['total_samples']
    ev['average_precision_rating'] = apr
    asr = success_sum / ev['total_samples']
    ev['average_success_rating'] = asr
    ev['average_score'] = (apr + asr) / 2.0
    ev['total_precision_rating'] = at20
    ev['total_success_rating'] = auc
    ev['total_score'] = (at20 + auc) / 2.0
    ev['probe_score'] = (apr + asr + at20 + auc) / 4.0
    ev['lost1'] = lost1
    ev['lost2'] = lost2
    ev['lost3'] = lost3
    ev['updates_max_frames'] = updates_max_frames
    ev['updates_confidence'] = updates_confidence
    ev['updates_total'] = updates_total
    evaluation_file = os.path.join(execution_dir, 'evaluation.txt')
    with open(evaluation_file, 'w') as f:
        for k, v in ev.items():
            f.write("{}={}\n".format(k, v))
    return ev
