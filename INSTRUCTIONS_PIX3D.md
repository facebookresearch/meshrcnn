# Experiments on Pix3D

## Download Pix3D and splits

Run

```
datasets/pix3d/download_pix3d.sh
```

to download [Pix3D][pix3d] and the `S1` & `S2` splits to `./datasets/pix3d/`

## Training

```
python tools/train_net.py --num-gpus 8 \
--config-file configs/pix3d/meshrcnn_R50_FPN.yaml
```

*Note* that the above config is tuned for 8-gpu distributed training.
Deviation from the provided training recipe means that other hyper parameters have to be tuned accordingly.

## Testing and Evaluation

```
python tools/train_net.py \
--config-file configs/pix3d/meshrcnn_R50_FPN.yaml \
--eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

If you wish to evaluate the provided pretrained models (see below for a model zoo), simply do `MODEL.WEIGHTS meshrcnn://meshrcnn_R50.pth`. *Note* that by default, the config files use the `S1` split.To change between `S1` and `S2`, specify the split in the `DATASETS` section in the config file.

## Models

We provide a model zoo for models trained on Pix3D `S1` & `S2` splits (see paper for more details).

|      |         Mesh R-CNN        |          Pixel2Mesh          |          SphereInit          |
|------|:-------------------------:|:----------------------------:|:----------------------------:|
| `S1` |   [meshrcnn_R50.pth][m1]  |   [pixel2mesh_R50.pth][pm1]  |   [sphereinit_R50.pth][sp1]  |
| `S2` | [meshrcnn_S2_R50.pth][m2] | [pixel2mesh_S2_R50.pth][pm2] | [sphereinit_S2_R50.pth][sp2] |

[pix3d]: http://pix3d.csail.mit.edu/data/pix3d.zip
[m1]: https://dl.fbaipublicfiles.com/meshrcnn/pix3d/meshrcnn_R50.pth
[m2]: https://dl.fbaipublicfiles.com/meshrcnn/pix3d/meshrcnn_S2_R50.pth
[pm1]: https://dl.fbaipublicfiles.com/meshrcnn/pix3d/pixel2mesh_R50.pth
[pm2]: https://dl.fbaipublicfiles.com/meshrcnn/pix3d/pixel2mesh_S2_R50.pth
[sp1]: https://dl.fbaipublicfiles.com/meshrcnn/pix3d/sphereinit_R50.pth
[sp2]: https://dl.fbaipublicfiles.com/meshrcnn/pix3d/sphereinit_S2_R50.pth
