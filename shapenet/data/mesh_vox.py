# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import os
import torch
from fvcore.common.file_io import PathManager
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image
from shapenet.data.utils import imagenet_preprocess
from shapenet.utils.coords import SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN, project_verts

logger = logging.getLogger(__name__)


class MeshVoxDataset(Dataset):
    def __init__(
        self,
        data_dir,
        normalize_images=True,
        split=None,
        return_mesh=False,
        voxel_size=32,
        num_samples=5000,
        sample_online=False,
        in_memory=False,
        return_id_str=False,
    ):

        super(MeshVoxDataset, self).__init__()
        if not return_mesh and sample_online:
            raise ValueError("Cannot sample online without returning mesh")
        self.data_dir = data_dir
        self.return_mesh = return_mesh
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.sample_online = sample_online
        self.return_id_str = return_id_str

        self.synset_ids = []
        self.model_ids = []
        self.image_ids = []
        self.mid_to_samples = {}

        transform = [T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        summary_json = os.path.join(data_dir, "summary.json")
        with PathManager.open(summary_json, "r") as f:
            summary = json.load(f)
            for sid in summary:
                logger.info("Starting synset %s" % sid)
                allowed_mids = None
                if split is not None:
                    if sid not in split:
                        logger.info("Skipping synset %s" % sid)
                        continue
                    elif isinstance(split[sid], list):
                        allowed_mids = set(split[sid])
                    elif isinstance(split, dict):
                        allowed_mids = set(split[sid].keys())
                for mid, num_imgs in summary[sid].items():
                    if allowed_mids is not None and mid not in allowed_mids:
                        continue
                    allowed_iids = None
                    if split is not None and isinstance(split[sid], dict):
                        allowed_iids = set(split[sid][mid])
                    if not sample_online and in_memory:
                        samples_path = os.path.join(data_dir, sid, mid, "samples.pt")
                        with PathManager.open(samples_path, "rb") as f:
                            samples = torch.load(f)
                        self.mid_to_samples[mid] = samples
                    for iid in range(num_imgs):
                        if allowed_iids is None or iid in allowed_iids:
                            self.synset_ids.append(sid)
                            self.model_ids.append(mid)
                            self.image_ids.append(iid)

    def __len__(self):
        return len(self.synset_ids)

    def __getitem__(self, idx):
        sid = self.synset_ids[idx]
        mid = self.model_ids[idx]
        iid = self.image_ids[idx]

        # Always read metadata for this model; TODO cache in __init__?
        metadata_path = os.path.join(self.data_dir, sid, mid, "metadata.pt")
        with PathManager.open(metadata_path, "rb") as f:
            metadata = torch.load(f)
        K = metadata["intrinsic"]
        RT = metadata["extrinsics"][iid]
        img_path = metadata["image_list"][iid]
        img_path = os.path.join(self.data_dir, sid, mid, "images", img_path)

        # Load the image
        with PathManager.open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = self.transform(img)

        # Maybe read mesh
        verts, faces = None, None
        if self.return_mesh:
            mesh_path = os.path.join(self.data_dir, sid, mid, "mesh.pt")
            with PathManager.open(mesh_path, "rb") as f:
                mesh_data = torch.load(f)
            verts, faces = mesh_data["verts"], mesh_data["faces"]
            verts = project_verts(verts, RT)

        # Maybe use cached samples
        points, normals = None, None
        if not self.sample_online:
            samples = self.mid_to_samples.get(mid, None)
            if samples is None:
                # They were not cached in memory, so read off disk
                samples_path = os.path.join(self.data_dir, sid, mid, "samples.pt")
                with PathManager.open(samples_path, "rb") as f:
                    samples = torch.load(f)
            points = samples["points_sampled"]
            normals = samples["normals_sampled"]
            idx = torch.randperm(points.shape[0])[: self.num_samples]
            points, normals = points[idx], normals[idx]
            points = project_verts(points, RT)
            normals = normals.mm(RT[:3, :3].t())  # Only rotate, don't translate

        voxels, P = None, None
        if self.voxel_size > 0:
            # Use precomputed voxels if we have them, otherwise return voxel_coords
            # and we will compute voxels in postprocess
            voxel_file = "vox%d/%03d.pt" % (self.voxel_size, iid)
            voxel_file = os.path.join(self.data_dir, sid, mid, voxel_file)
            if PathManager.isfile(voxel_file):
                with PathManager.open(voxel_file, "rb") as f:
                    voxels = torch.load(f)
            else:
                voxel_path = os.path.join(self.data_dir, sid, mid, "voxels.pt")
                with PathManager.open(voxel_path, "rb") as f:
                    voxel_data = torch.load(f)
                voxels = voxel_data["voxel_coords"]
                P = K.mm(RT)

        id_str = "%s-%s-%02d" % (sid, mid, iid)
        return img, verts, faces, points, normals, voxels, P, id_str

    def _voxelize(self, voxel_coords, P):
        V = self.voxel_size
        device = voxel_coords.device
        voxel_coords = project_verts(voxel_coords, P)

        # In the original coordinate system, the object fits in a unit sphere
        # centered at the origin. Thus after transforming by RT, it will fit
        # in a unit sphere centered at T = RT[:, 3] = (0, 0, RT[2, 3]). We need
        # to figure out what the range of z will be after being further
        # transformed by K. We can work this out explicitly.
        # z0 = RT[2, 3].item()
        # zp, zm = z0 - 0.5, z0 + 0.5
        # k22, k23 = K[2, 2].item(), K[2, 3].item()
        # k32, k33 = K[3, 2].item(), K[3, 3].item()
        # zmin = (zm * k22 + k23) / (zm * k32 + k33)
        # zmax = (zp * k22 + k23) / (zp * k32 + k33)

        # Using the actual zmin and zmax of the model is bad because we need them
        # to perform the inverse transform, which transform voxels back into world
        # space for refinement or evaluation. Instead we use an empirical min and
        # max over the dataset; that way it is consistent for all images.
        zmin = SHAPENET_MIN_ZMIN
        zmax = SHAPENET_MAX_ZMAX

        # Once we know zmin and zmax, we need to adjust the z coordinates so the
        # range [zmin, zmax] instead runs from [-1, 1]
        m = 2.0 / (zmax - zmin)
        b = -2.0 * zmin / (zmax - zmin) - 1
        voxel_coords[:, 2].mul_(m).add_(b)
        voxel_coords[:, 1].mul_(-1)  # Flip y

        # Now voxels are in [-1, 1]^3; map to [0, V-1)^3
        voxel_coords = 0.5 * (V - 1) * (voxel_coords + 1.0)
        voxel_coords = voxel_coords.round().to(torch.int64)
        valid = (0 <= voxel_coords) * (voxel_coords < V)
        valid = valid[:, 0] * valid[:, 1] * valid[:, 2]
        x, y, z = voxel_coords.unbind(dim=1)
        x, y, z = x[valid], y[valid], z[valid]
        voxels = torch.zeros(V, V, V, dtype=torch.int64, device=device)
        voxels[z, y, x] = 1

        return voxels

    @staticmethod
    def collate_fn(batch):
        imgs, verts, faces, points, normals, voxels, Ps, id_strs = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        if verts[0] is not None and faces[0] is not None:
            # TODO(gkioxari) Meshes should accept tuples
            meshes = Meshes(verts=list(verts), faces=list(faces))
        else:
            meshes = None
        if points[0] is not None and normals[0] is not None:
            points = torch.stack(points, dim=0)
            normals = torch.stack(normals, dim=0)
        else:
            points, normals = None, None
        if voxels[0] is None:
            voxels = None
            Ps = None
        elif voxels[0].dim() == 2:
            # They are voxel coords
            Ps = torch.stack(Ps, dim=0)
        elif voxels[0].dim() == 3:
            # They are actual voxels
            voxels = torch.stack(voxels, dim=0)
        return imgs, meshes, points, normals, voxels, Ps, id_strs

    def postprocess(self, batch, device=None):
        if device is None:
            device = torch.device("cuda")
        imgs, meshes, points, normals, voxels, Ps, id_strs = batch
        imgs = imgs.to(device)
        if meshes is not None:
            meshes = meshes.to(device)
        if points is not None and normals is not None:
            points = points.to(device)
            normals = normals.to(device)
        else:
            points, normals = sample_points_from_meshes(
                meshes, num_samples=self.num_samples, return_normals=True
            )
        if voxels is not None:
            if torch.is_tensor(voxels):
                # We used cached voxels on disk, just cast and return
                voxels = voxels.to(device)
            else:
                # We got a list of voxel_coords, and need to compute voxels on-the-fly
                voxel_coords = voxels
                Ps = Ps.to(device)
                voxels = []
                for i, cur_voxel_coords in enumerate(voxel_coords):
                    cur_voxel_coords = cur_voxel_coords.to(device)
                    cur_voxels = self._voxelize(cur_voxel_coords, Ps[i])
                    voxels.append(cur_voxels)
                voxels = torch.stack(voxels, dim=0)

        if self.return_id_str:
            return imgs, meshes, points, normals, voxels, id_strs
        else:
            return imgs, meshes, points, normals, voxels
