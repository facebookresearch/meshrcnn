# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import os
from pytorch3d.io import save_obj
from termcolor import colored

from tabulate import tabulate

try:
    import cv2  # noqa
except ImportError:
    # If opencv is not available, everything else should still run
    pass


def print_instances_class_histogram(num_instances, class_names, results):
    """
    Args:
        num_instances (list): list of dataset dicts.
    """
    num_classes = len(class_names)
    N_COLS = 7
    data = list(
        itertools.chain(
            *[
                [
                    class_names[id],
                    v,
                    results["chamfer"][id] / v,
                    results["normal"][id] / v,
                    results["f1_01"][id] / v,
                    results["f1_03"][id] / v,
                    results["f1_05"][id] / v,
                ]
                for id, v in num_instances.items()
            ]
        )
    )
    total_num_instances = sum(data[1::7])
    mean_chamfer = sum(data[2::7]) / num_classes
    mean_normal = sum(data[3::7]) / num_classes
    mean_f1_01 = sum(data[4::7]) / num_classes
    mean_f1_03 = sum(data[5::7]) / num_classes
    mean_f1_05 = sum(data[6::7]) / num_classes
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data.extend(
        [
            "total",
            total_num_instances,
            mean_chamfer,
            mean_normal,
            mean_f1_01,
            mean_f1_03,
            mean_f1_05,
        ]
    )
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data.extend(
        [
            "per-instance",
            total_num_instances,
            sum(results["chamfer"].values()) / total_num_instances,
            sum(results["normal"].values()) / total_num_instances,
            sum(results["f1_01"].values()) / total_num_instances,
            sum(results["f1_03"].values()) / total_num_instances,
            sum(results["f1_05"].values()) / total_num_instances,
        ]
    )
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances", "chamfer", "normal", "F1(0.1)", "F1(0.3)", "F1(0.5)"]
        * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(
        "Distribution of testing instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan")
    )


def print_instances_class_histogram_p2m(num_instances, class_names, results):
    """
    Args:
        num_instances (list): list of dataset dicts.
    """
    num_classes = len(class_names)
    N_COLS = 6
    data = list(
        itertools.chain(
            *[
                [
                    class_names[id],
                    v,
                    results["chamfer"][id] / v,
                    results["normal"][id] / v,
                    results["f1_1e_4"][id] / v,
                    results["f1_2e_4"][id] / v,
                ]
                for id, v in num_instances.items()
            ]
        )
    )
    total_num_instances = sum(data[1::6])
    mean_chamfer = sum(data[2::6]) / num_classes
    mean_normal = sum(data[3::6]) / num_classes
    mean_f1_1e_4 = sum(data[4::6]) / num_classes
    mean_f1_2e_4 = sum(data[5::6]) / num_classes
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data.extend(
        ["total", total_num_instances, mean_chamfer, mean_normal, mean_f1_1e_4, mean_f1_2e_4]
    )
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data.extend(
        [
            "per-instance",
            total_num_instances,
            sum(results["chamfer"].values()) / total_num_instances,
            sum(results["normal"].values()) / total_num_instances,
            sum(results["f1_1e_4"].values()) / total_num_instances,
            sum(results["f1_2e_4"].values()) / total_num_instances,
        ]
    )
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances", "chamfer", "normal", "F1(0.0001)", "F1(0.0002)"]
        * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(
        "Distribution of testing instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan")
    )


def visualize_prediction(image_id, img, mesh, output_dir):
    # create vis_dir
    output_dir = os.path.join(output_dir, "results_shapenet")
    os.makedirs(output_dir, exist_ok=True)

    save_img = os.path.join(output_dir, "%s.png" % (image_id))
    cv2.imwrite(save_img, img[:, :, ::-1])

    save_mesh = os.path.join(output_dir, "%s.obj" % (image_id))
    verts, faces = mesh.get_mesh_verts_faces(0)
    save_obj(save_mesh, verts, faces)
