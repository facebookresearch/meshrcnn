# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

logger = logging.getLogger(__name__)


class MeshLoss(nn.Module):
    def __init__(
        self,
        chamfer_weight=1.0,
        normal_weight=0.0,
        edge_weight=0.1,
        voxel_weight=0.0,
        gt_num_samples=5000,
        pred_num_samples=5000,
    ):

        super(MeshLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.gt_num_samples = gt_num_samples
        self.pred_num_samples = pred_num_samples
        self.voxel_weight = voxel_weight

        self.skip_mesh_loss = False
        if chamfer_weight == 0.0 and normal_weight == 0.0 and edge_weight == 0.0:
            self.skip_mesh_loss = True

    def forward(self, voxel_scores, meshes_pred, voxels_gt, meshes_gt):
        """
        Args:
          meshes_pred: Meshes
          meshes_gt: Either Meshes, or a tuple (points_gt, normals_gt)

        Returns:
          loss (float): Torch scalar giving the total loss, or None if an error occured and
                we should skip this loss. TODO use an exception instead?
          losses (dict): A dictionary mapping loss names to Torch scalars giving their
                        (unweighted) values.
        """
        # Sample from meshes_gt if we haven't already
        if isinstance(meshes_gt, tuple):
            points_gt, normals_gt = meshes_gt
        else:
            points_gt, normals_gt = sample_points_from_meshes(
                meshes_gt, num_samples=self.gt_num_samples, return_normals=True
            )

        total_loss = torch.tensor(0.0).to(points_gt)
        losses = {}

        if voxel_scores is not None and voxels_gt is not None and self.voxel_weight > 0:
            voxels_gt = voxels_gt.float()
            voxel_loss = F.binary_cross_entropy_with_logits(voxel_scores, voxels_gt)
            total_loss = total_loss + self.voxel_weight * voxel_loss
            losses["voxel"] = voxel_loss

        if isinstance(meshes_pred, Meshes):
            meshes_pred = [meshes_pred]
        elif meshes_pred is None:
            meshes_pred = []

        # Now assume meshes_pred is a list
        if not self.skip_mesh_loss:
            for i, cur_meshes_pred in enumerate(meshes_pred):
                cur_out = self._mesh_loss(cur_meshes_pred, points_gt, normals_gt)
                cur_loss, cur_losses = cur_out
                if total_loss is None or cur_loss is None:
                    total_loss = None
                else:
                    total_loss = total_loss + cur_loss / len(meshes_pred)
                for k, v in cur_losses.items():
                    losses["%s_%d" % (k, i)] = v

        return total_loss, losses

    def _mesh_loss(self, meshes_pred, points_gt, normals_gt):
        """
        Args:
          meshes_pred: Meshes containing N meshes
          points_gt: Tensor of shape NxPx3
          normals_gt: Tensor of shape NxPx3

        Returns:
          total_loss (float): The sum of all losses specific to meshes
          losses (dict): All (unweighted) mesh losses in a dictionary
        """
        zero = torch.tensor(0.0).to(meshes_pred.verts_list()[0])
        losses = {"chamfer": zero, "normal": zero, "edge": zero}
        points_pred, normals_pred = sample_points_from_meshes(
            meshes_pred, num_samples=self.pred_num_samples, return_normals=True
        )

        total_loss = torch.tensor(0.0).to(points_pred)
        if points_pred is None or points_gt is None:
            # Sampling failed, so return None
            total_loss = None
            which = "predictions" if points_pred is None else "GT"
            logger.info("WARNING: Sampling %s failed" % (which))
            return total_loss, losses

        losses = {}
        cham_loss, normal_loss = chamfer_distance(points_pred, points_gt, normals_pred, normals_gt)

        total_loss = total_loss + self.chamfer_weight * cham_loss
        total_loss = total_loss + self.normal_weight * normal_loss
        losses["chamfer"] = cham_loss
        losses["normal"] = normal_loss

        edge_loss = mesh_edge_loss(meshes_pred)
        total_loss = total_loss + self.edge_weight * edge_loss
        losses["edge"] = edge_loss

        return total_loss, losses
