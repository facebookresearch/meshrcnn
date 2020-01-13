# Experiments on ShapeNet

## Data

We use [ShapeNet][shapenet] data and their renderings, as provided by [R2N2][r2n2].

Run
```
datasets/shapenet/download_shapenet.sh
```

The above command downloads [R2N2][r2n2] and the train/val/test splits.
You need to place ShapeNet Core v1 (after registering and downloading) in `datasets/shapenet/`

## Preprocessing

    ./tools/preprocess_shapenet.py \
    --r2n2_dir ./datasets/shapenet/ShapeNetRendering \
    --shapenet_dir ./datasets/shapenet/ShapeNetCore.v1 \
    --output_dir ./datasets/shapenet/ShapeNetV1processed \
    --zip_output


The above command preprocesses the ShapeNet dataset to reduce the data loading time.
The preprocessed data will be saved in `output_dir` and will be zipped.
The zipped output is useful when training in clusters.

## Training

```
./tools/train_net_shapenet.py --num-gpus 8 --config-file configs/shapenet/voxmesh_R_50.yaml
```

When `--copy_data`, the preprocessed zipped data from above will be copied to a local `/tmp` directory.
This is particularly useful when training on remote clusters, as it reduces the io time during training

## Testing and Evaluation


    ./tools/train_net_shapenet.py --eval-only --num-gpus 1 \
    --config-file configs/shapenet/voxmesh_R_50.yaml \
    MODEL.WEIGHTS shapenet://voxmesh_R50.pth

The output of the evaluation produces the results as shown in Table 2 of our paper, meaning it scale normalizes the ground truth meshes.
To evaluate under the [Pixel2Mesh][p2m] protocol, as in Table 1 of our paper, add `--eval-p2m`.

## Models

We provide pretrained models for our approach

[shapenet]: http://shapenet.cs.stanford.edu/
[r2n2_data]: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
[r2n2]: http://3d-r2n2.stanford.edu/
[p2m]: https://github.com/nywang16/Pixel2Mesh
