# Experiments on ShapeNet

## Data

We use [ShapeNet][shapenet] data and their renderings, as provided by [R2N2][r2n2].

Run

```
datasets/shapenet/download_shapenet.sh
```

to download [R2N2][r2n2], and the train/val/test splits.
You also need the original ShapeNet Core v1 & binvox dataset, which require [registration][shapenet_login] before downloading.

## Preprocessing

```
python tools/preprocess_shapenet.py \
--shapenet_dir /path/to/ShapeNetCore.v1 \
--shapenet_binvox_dir /path/to/ShapeNetCore.v1.binvox \
--output_dir ./datasets/shapenet/ShapeNetV1processed \
--zip_output
```

The above command preprocesses the ShapeNet dataset to reduce the data loading time.
The preprocessed data will be saved in `./datasets/shapenet` and will be zipped.
The zipped output is useful when training in clusters.

## Training

```
python tools/train_net_shapenet.py --num-gpus 8 \
--config-file configs/shapenet/voxmesh_R50.yaml
```

When `--copy_data`, the preprocessed zipped data from above will be copied to a local `/tmp` directory.
This is particularly useful when training on remote clusters, as it reduces the io time during training

## Testing and Evaluation

```
python tools/train_net_shapenet.py --eval-only --num-gpus 1 \
--config-file configs/shapenet/voxmesh_R50.yaml \
MODEL.CHECKPOINT shapenet://voxmesh_R50.pth
```

The output of the evaluation produces the results as shown in Table 2 of our paper.
To evaluate under the [Pixel2Mesh][p2m] protocol, as in Table 1 of our paper, add `--eval-p2m`.

## Models

|          Mesh R-CNN          |          Pixel2Mesh          |          SphereInit          |
|-----------------------------:|:----------------------------:|:----------------------------:|
|  [voxmesh_R50.pth][voxm]     |   [pixel2mesh_R50.pth][pm]   |   [sphereinit_R50.pth][sp]   |

Note that we release only the *light* and *pretty* for both our and the baseline models.

## Performance

### Scale-normalized Protocol

Performance of our [model][voxm] on ShapeNet `test` set under the scale-normalized evaluation protocol (as in Table 2 of our paper).

|   category   | #instances   | chamfer   | normal   | F1(0.1)   | F1(0.3)   | F1(0.5)   |
|:------------:|:-------------|:----------|:---------|:----------|:----------|:----------|
|    bench     | 8712         | 0.120899  | 0.657536 | 42.4005   | 86.0036   | 95.128    |
|    chair     | 32520        | 0.183693  | 0.712362 | 31.6906   | 79.8275   | 92.0139   |
|     lamp     | 11122        | 0.413965  | 0.672992 | 30.5048   | 70.3449   | 84.5068   |
|   speaker    | 7752         | 0.253796  | 0.730829 | 24.8335   | 74.6606   | 88.237    |
|   firearm    | 11386        | 0.168323  | 0.621439 | 47.2251   | 85.271    | 93.8171   |
|    table     | 40796        | 0.148357  | 0.75642  | 42.249    | 86.2039   | 94.1623   |
|  watercraft  | 9298         | 0.224168  | 0.642812 | 30.0589   | 75.5332   | 89.9764   |
|    plane     | 19416        | 0.187465  | 0.684285 | 39.009    | 80.998    | 92.1069   |
|   cabinet    | 7541         | 0.111294  | 0.75122  | 34.8227   | 86.9346   | 95.371    |
|     car      | 35981        | 0.107605  | 0.647857 | 29.6397   | 85.7925   | 96.2938   |
|   monitor    | 5256         | 0.218032  | 0.779365 | 27.2531   | 77.2979   | 90.904    |
|    couch     | 15226        | 0.144279  | 0.72302  | 27.5734   | 81.684    | 94.3294   |
|  cellphone   | 5045         | 0.121504  | 0.850437 | 42.9168   | 88.9888   | 96.1367   |
|              |              |           |          |           |           |           |
|    total     | 210051       | 0.184875  | 0.710044 | 34.629    | 81.5031   | 92.5372   |
|              |              |           |          |           |           |           |
| per-instance | 210051       | 0.171189  | 0.70275  | 34.9372   | 82.4107   | 93.1323   |


### Pixel2Mesh Protocol

Performance of our [model][voxm] on ShapeNet `test` set under the pixel2mesh evaluation protocol (as in Table 1 of our paper). To evaluate under this protocol, add the `--eval-p2m` flag.

|   category   | #instances   | chamfer     | normal   | F1(0.0001)   | F1(0.0002)   |
|:------------:|:-------------|:------------|:---------|:-------------|:-------------|
|    bench     | 8712         | 0.000295252 | 0.657508 | 73.4681      | 84.4999      |
|    chair     | 32520        | 0.000400415 | 0.712348 | 66.5227      | 79.3634      |
|     lamp     | 11122        | 0.000788915 | 0.673057 | 60.1057      | 71.7711      |
|   speaker    | 7752         | 0.000582152 | 0.730797 | 59.9974      | 73.8792      |
|   firearm    | 11386        | 0.000357016 | 0.621438 | 75.9761      | 85.5111      |
|    table     | 40796        | 0.000342991 | 0.756442 | 76.0776      | 85.4878      |
|  watercraft  | 9298         | 0.000449061 | 0.642791 | 62.808       | 76.5464      |
|    plane     | 19416        | 0.000313141 | 0.684333 | 75.8104      | 85.3897      |
|   cabinet    | 7541         | 0.000293613 | 0.751306 | 72.6302      | 84.7327      |
|     car      | 35981        | 0.000240585 | 0.647896 | 71.8118      | 85.5155      |
|   monitor    | 5256         | 0.000470965 | 0.779397 | 64.2917      | 77.8422      |
|    couch     | 15226        | 0.000355369 | 0.723013 | 64.1388      | 79.327       |
|  cellphone   | 5045         | 0.000280397 | 0.850456 | 77.3011      | 87.8698      |
|              |              |             |          |              |              |
|    total     | 210051       | 0.000397682 | 0.71006  | 69.303       | 81.3643      |
|              |              |             |          |              |              |
| per-instance | 210051       | 0.000368317 | 0.702768 | 70.4479      | 82.3373      |


[shapenet]: http://shapenet.cs.stanford.edu/
[shapenet_login]: https://www.shapenet.org/login/
[r2n2_data]: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
[r2n2]: http://3d-r2n2.stanford.edu/
[p2m]: https://github.com/nywang16/Pixel2Mesh
[voxm]: https://dl.fbaipublicfiles.com/meshrcnn/shapenet/voxmesh_R50.pth
[pm]: https://dl.fbaipublicfiles.com/meshrcnn/shapenet/pixel2mesh_R50.pth
[sp]: https://dl.fbaipublicfiles.com/meshrcnn/shapenet/sphereinit_R50.pth

<!---
voxmesh: f162820673
pixel2mesh: f163108637
sphereinit: f163113815
-->
