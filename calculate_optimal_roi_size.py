import os
import itertools
import numpy as np
from core.Rect import Rect
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import yaml
import zipfile
import math


env_config_path = "./config/environment.yaml"
tracker_config_path = "./config/tracker.yaml"


def get_distances_and_sizes(file_handle):
    dists = []
    rads = []
    last_pos = None
    for line in file_handle:
        pos = Rect(line.decode().split(','))
        if last_pos is not None:
            dists.append(abs(pos.center_distance(last_pos)))
        rads.append(max(pos.width, pos.height) / 2)
        #rads.append(math.sqrt(max(pos.width ** 2, pos.height ** 2)))
        last_pos = pos
    return dists, rads


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def avg(iterable):
    return sum(iterable) / len(iterable)


if __name__ == '__main__':
    with open(env_config_path, 'r') as env_config_file:
        env_config = yaml.load(env_config_file)
    with open(tracker_config_path, 'r') as tracker_config_file:
        tracker_config = yaml.load(tracker_config_file)

    results = {
        "total": {
            "distances": [],
            "radii": []
        },
        "individual": []
    }
    for sample in tracker_config["tracking"]:
        with zipfile.ZipFile(os.path.join(env_config["data_dir"], sample + ".zip")) as zip_file:
            with zip_file.open("groundtruth_rect.txt", 'r') as gt_file:
                result = get_distances_and_sizes(gt_file)
                #print(str(result[0]))
                #print(str(result[1]))
                #print("----------------------")
                results["individual"].append({
                    "name": sample,
                    "distances": sorted(result[0], reverse=True),
                    "radii": sorted(result[1], reverse=True)
                })

    for sample in results["individual"]:
        results["total"]["distances"].extend(sample["distances"])
        results["total"]["radii"].extend(sample["radii"])
    results["total"]["distances"].sort(reverse=True)
    results["total"]["radii"].sort(reverse=True)

    print("biggest distances: [{}]".format(results["total"]["distances"][0:10]))
    print("Average distance: {}".format(avg(results["total"]["distances"])))
    print("biggest radii: [{}]".format(results["total"]["radii"][0:10]))
    print("Average radius: {}".format(avg(results["total"]["radii"])))
    print("Samples with biggest avg distances: {}".format(
        ["{} ({})".format(sample["name"], avg(sample["distances"])) for sample in
         sorted(results["individual"], reverse=True,
             key=lambda sample: avg(sample["distances"]))[:10]]
    ))
    print("Samples with biggest max distances: {}".format(
        ["{} ({})".format(sample["name"], sample["distances"][0]) for sample in
         sorted(results["individual"], reverse=True,
             key=lambda sample: sample["distances"][0])[:10]]
    ))
    print("Samples with biggest avg sizes: {}".format(
        ["{} ({})".format(sample["name"], avg(sample["radii"])) for sample in
         sorted(results["individual"], reverse=True,
             key=lambda sample: avg(sample["radii"]))[:10]]
    ))
    print("Samples with biggest max sizes: {}".format(
        ["{} ({})".format(sample["name"], sample["radii"][0]) for sample in
         sorted(results["individual"], reverse=True,
             key=lambda sample: sample["radii"][0])[:10]]
    ))

    print("recommended sroi size:")
    print("    (max_distance * 2 + max_size / 2) * 1.2'")
    print(" => ({} * 2 + {} / 2) * 1.2".format(
        results["total"]["distances"][0],
        results["total"]["radii"][0]))
    print(" => {}".format(
        ((results["total"]["distances"][0] * 2)
         + (results["total"]["radii"][0] / 2)) * 1.2))


    output_dir = os.path.join(env_config["data_dir"], tracker_config["tracking"][0].split('/', 1)[0])

    pyplot.figure()
    pyplot.boxplot(np.array(results["total"]["distances"]))
    pyplot.text(0.05, 0.05, "total distances")
    pyplot.show()
    pyplot.close()

    pyplot.figure()
    pyplot.boxplot(np.array(results["total"]["radii"]))
    pyplot.text(0.05, 0.05, "total sizes")
    pyplot.show()
    pyplot.close()
