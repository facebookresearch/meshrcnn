# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import itertools
import numpy as np
import os
import pycocotools.mask as mask_util
from detectron2.data import detection_utils
from detectron2.structures import Boxes, BoxMode
from detectron2.utils.colormap import colormap
from pytorch3d.io import save_obj
from pytorch3d.ops import sample_points_from_meshes
from termcolor import colored

from tabulate import tabulate

try:
    import cv2  # noqa
except ImportError:
    # If opencv is not available, everything else should still run
    pass


"""
This module contains some common visualization utilities.
It plots text/boxes/masks/keypoints on an image, with some pre-defined "artistic" style & color.

These functions expect BGR images in (H, W, 3), with values in range [0, 255].
They all return an uint8 image of shape (H, W, 3), and the function may modify the input in-place.
"""


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def print_instances_class_histogram(num_instances, class_names, results):
    """
    Args:
        num_instances (list): list of dataset dicts.
        class_names (list): list of class_names
    """
    num_classes = len(class_names)
    N_COLS = 5
    data = list(
        itertools.chain(
            *[
                [
                    class_names[i],
                    int(v),
                    results["box_ap@0.5 - %s" % (class_names[i])],
                    results["mask_ap@0.5 - %s" % (class_names[i])],
                    results["mesh_ap@0.5 - %s" % (class_names[i])],
                ]
                for i, v in enumerate(num_instances)
            ]
        )
    )
    total_num_instances = sum(data[1::5])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data.extend(
        [
            "total",
            total_num_instances,
            results["box_ap@0.5"],
            results["mask_ap@0.5"],
            results["mesh_ap@0.5"],
        ]
    )
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances", "boxAP", "maskAP", "meshAP"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(
        "Distribution of testing instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan")
    )


def draw_text(img, pos, text, font_scale=0.35):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.2 * text_h) < 0:
        y0 = int(1.2 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, _GREEN, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.2 * text_h)
    cv2.putText(img, text, text_bottomleft, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def draw_boxes(img, boxes, thickness=1):
    """
    Draw boxes on an image.

    Args:
        boxes (Boxes or ndarray): either a :class:`Boxes` instances,
            or a Nx4 numpy array of XYXY_ABS format.
        thickness (int): the thickness of the edges
    """
    img = img.astype(np.uint8)
    if isinstance(boxes, Boxes):
        boxes = boxes.clone("xyxy")
    else:
        assert boxes.ndim == 2, boxes.shape
    for box in boxes:
        (x0, y0, x1, y1) = (int(x + 0.5) for x in box)
        img = cv2.rectangle(img, (x0, y0), (x1, y1), color=_GREEN, thickness=thickness)
    return img


def draw_mask(img, mask, color, alpha=0.4, draw_contours=True):
    """
    Draw (overlay) a mask on an image.

    Args:
        mask (ndarray): an (H, W) array of the same spatial size as the image.
            Nonzero positions in the array are considered part of the mask.
        color: a BGR color
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        draw_contours (bool): whether to also draw the contours of every
            connected component (object part) in the mask.
    """
    img = img.astype(np.float32)

    idx = np.nonzero(mask)
    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * color

    if draw_contours:
        # opencv func signature has changed between versions
        contours = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(img, contours, -1, _WHITE, 1, cv2.LINE_AA)
    return img.astype(np.uint8)


def draw_keypoints(img, keypoints):
    """
    Args:
        keypoints (ndarray): Nx2 array, each row is an (x, y) coordinate.
    """
    for coord in keypoints:
        cv2.circle(img, tuple(coord), thickness=-1, lineType=cv2.LINE_AA, radius=3, color=_GREEN)
    return img


def draw_pix3d_dict(dataset_dict, class_names=None):
    """
    Draw the instance annotations for an image.

    Args:
        dataset_dict (dict): a dict in Detectron2 Dataset format. See DATASETS.md
        class_names (list[str] or None): `class_names[cateogory_id]` is the
            name for this category. If not provided, the visualization will
            not contain class names.
    """
    img = dataset_dict.get("image", None)
    if img is None:
        img = cv2.imread(dataset_dict["file_name"])
    annos = dataset_dict["annotations"]
    if not len(annos):
        return img
    boxes = np.asarray(
        [BoxMode.convert(k["bbox"], k["bbox_mode"], BoxMode.XYXY_ABS) for k in annos]
    )

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    sorted_boxes = copy.deepcopy(boxes[sorted_inds])

    img = draw_boxes(img, sorted_boxes)

    cmap = colormap()

    for num, i in enumerate(sorted_inds):
        anno = annos[i]
        bbox = anno["bbox"]
        assert anno["bbox_mode"] in [
            BoxMode.XYXY_ABS,
            BoxMode.XYWH_ABS,
        ], "Relative coordinates not yet supported in visualization."
        iscrowd = anno.get("iscrowd", 0)
        clsid = anno["category_id"]
        text = class_names[clsid] if class_names is not None else str(clsid)
        if iscrowd:
            text = text + "_crowd"
        img = draw_text(img, (bbox[0], bbox[1] - 2), text)

        segs = anno.get("segmentation", None)
        if segs is not None and not iscrowd:
            segs_color = cmap[num % len(cmap)]
            mask = cv2.imread(segs)
            img = draw_mask(img, mask, segs_color, draw_contours=False)

        kpts = anno.get("keypoints", None)
        if kpts is not None and not iscrowd:
            kpts = np.asarray(kpts).reshape(-1, 3)[:, :2]
            img = draw_keypoints(img, kpts)
    return img


# minibatch visualization of boxes, masks and targets
def visualize_minibatch(images, data, output_dir, vis_fg=False):
    import matplotlib.pyplot as plt

    # create vis_dir
    output_dir = os.path.join(output_dir, "minibatch")
    os.makedirs(output_dir, exist_ok=True)

    # read image and add mean
    ims = images.tensor.cpu().numpy()
    num = ims.shape[0]
    pixel_mean = np.expand_dims(np.expand_dims(np.array([102.9801, 115.9465, 122.7717]), 1), 2)

    proposals = data["proposals"]
    if vis_fg:
        proposals = data["fg_proposals"]
    assert len(proposals) == num

    index = np.array([prop.proposal_boxes.tensor.shape[0] for prop in proposals]).cumsum()
    index = np.concatenate((np.array([0]), index[:-1]))

    if "target_meshes" in data:
        target_sampled_verts = sample_points_from_meshes(
            data["target_meshes"], num_samples=10000, return_normals=False
        )

    for i in range(num):
        im = ims[i, :, :, :] + pixel_mean
        im = im.transpose(1, 2, 0)[:, :, (2, 1, 0)]
        im = im.astype(np.uint8, copy=False)
        boxes = proposals[i].proposal_boxes.tensor.cpu().numpy()
        gt_classes = proposals[i].gt_classes.cpu().numpy()

        for j in range(boxes.shape[0]):
            fig = plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(im)
            roi = boxes[j]
            plt.gca().add_patch(
                plt.Rectangle(
                    (roi[0], roi[1]),
                    roi[2] - roi[0],
                    roi[3] - roi[1],
                    fill=False,
                    edgecolor="r",
                    linewidth=3,
                )
            )
            plt.title("Class %d" % (gt_classes[j]))
            if vis_fg and "target_masks" in data:
                mask = proposals[i].gt_masks[j].cpu().numpy()
                plt.subplot(2, 3, 2)
                plt.imshow(mask)
                plt.gca().add_patch(
                    plt.Rectangle(
                        (roi[0], roi[1]),
                        roi[2] - roi[0],
                        roi[3] - roi[1],
                        fill=False,
                        edgecolor="g",
                        linewidth=3,
                    )
                )
                mask = data["target_masks"][index[i] + j].cpu().numpy()
                plt.subplot(2, 3, 3)
                plt.imshow(mask)
                plt.title("Mask")
            if vis_fg and "target_voxels" in data:
                voxel = data["target_voxels"][index[i] + j].cpu().numpy()  # (D, H, W)
                voxel = voxel.transpose(1, 2, 0)
                plt.subplot(2, 3, 4)
                plt.imshow(np.max(voxel, 2))
                plt.title("Voxel")
            if vis_fg and "target_meshes" in data:
                resolution = 28
                verts = target_sampled_verts[int(index[i] + j)]
                verts = verts.cpu().numpy()
                x = (verts[:, 0] + 1) * (resolution - 1) / 2.0
                y = (verts[:, 1] + 1) * (resolution - 1) / 2.0
                x_keep = np.logical_and(x >= 0, x < resolution)
                y_keep = np.logical_and(y >= 0, y < resolution)
                keep = np.logical_and(x_keep, y_keep)
                x = x[keep].astype(np.int32)
                y = y[keep].astype(np.int32)
                target_img = np.zeros((resolution, resolution), dtype=np.uint8)
                target_img[y, x] = 255
                plt.subplot(2, 3, 5)
                plt.imshow(target_img)
                plt.title("Mesh")
            if vis_fg and "init_meshes" in data:
                resolution = 28
                verts, faces = data["init_meshes"].get_mesh_verts_faces(int(index[i] + j))
                verts = verts.cpu().numpy()
                x = (verts[:, 0] + 1) * (resolution - 1) / 2.0
                y = (verts[:, 1] + 1) * (resolution - 1) / 2.0
                x_keep = np.logical_and(x >= 0, x < resolution)
                y_keep = np.logical_and(y >= 0, y < resolution)
                keep = np.logical_and(x_keep, y_keep)
                x = x[keep].astype(np.int32)
                y = y[keep].astype(np.int32)
                target_img = np.zeros((resolution, resolution), dtype=np.uint8)
                target_img[y, x] = 255
                plt.subplot(2, 3, 6)
                plt.imshow(target_img)
                plt.title("Init Mesh")
            if not vis_fg:
                mask = proposals[i].gt_masks[j].cpu().numpy()
                plt.subplot(1, 4, 2)
                plt.imshow(mask)
                plt.gca().add_patch(
                    plt.Rectangle(
                        (roi[0], roi[1]),
                        roi[2] - roi[0],
                        roi[3] - roi[1],
                        fill=False,
                        edgecolor="g",
                        linewidth=3,
                    )
                )

            save_file = os.path.join(
                output_dir, format(np.random.randint(10000)) + "_" + format(j) + ".png"
            )
            plt.savefig(save_file)
            plt.close(fig)


def visualize_predictions(
    image_id,
    image_file,
    scores,
    labels,
    boxes,
    mask_rles,
    meshes,
    metadata,
    output_dir,
    alpha=0.6,
    dpi=200,
):

    # create vis_dir
    output_dir = os.path.join(output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    cat_colors = metadata.thing_colors
    cat_names = metadata.thing_classes

    # read image
    image_file = os.path.join(metadata.image_root, image_file)
    image = detection_utils.read_image(image_file, format="RGB")

    num_preds = len(scores)

    for i in range(num_preds):
        # box
        box = boxes[i].view(1, 4)
        # RLE to 2D mask
        mask = mask_util.decode(mask_rles[i])

        label = labels[i]
        mask_color = np.array(cat_colors[label], dtype=np.float32)
        cat_name = cat_names[label]
        score = scores[i]

        # plot mask overlayed
        composite = image.copy()
        composite = draw_mask(composite, mask, mask_color, alpha=alpha, draw_contours=False)
        thickness = int(np.ceil(0.001 * image.shape[0]))
        composite = draw_boxes(composite, box, thickness)

        save_file = os.path.join(output_dir, "%d_%d_%s_%.3f.png" % (image_id, i, cat_name, score))
        cv2.imwrite(save_file, composite[:, :, ::-1])

        save_file = os.path.join(output_dir, "%d_%d_%s_%.3f.obj" % (image_id, i, cat_name, score))
        verts, faces = meshes.get_mesh_verts_faces(i)
        save_obj(save_file, verts, faces)
