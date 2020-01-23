# Mesh R-CNN

Code for the paper

**[Mesh R-CNN][1]**  
[Georgia Gkioxari][gg], Jitendra Malik, [Justin Johnson][jj]  
ICCV 2019

## Installation Requirements
- [Detectron2][d2]
- [Pytorch3D][py3d]

The implementation of Mesh R-CNN is based on [detectron2][d2] and [pytorch3d][py3d].
You will first need to install those in order to be able to run Mesh R-CNN.

To install
```
git clone https://github.com/facebookresearch/meshrcnn.git
cd meshrcnn && pip install -e .
```

## Demo

Run Mesh R-CNN on an input image

```
python demo/demo.py \
--config-file configs/pix3d/meshrcnn_R50_FPN.yaml \
--input /path/to/image \
--output output_demo \
--onlyhighest MODEL.WEIGHTS meshrcnn://meshrcnn_R50.pth
```

See [demo.py](demo/demo.py) for more details.

## Running Experiments

### Pix3D
See [INSTRUCTIONS_PIX3D.md](INSTRUCTIONS_PIX3D.md) for more instructions.

### ShapeNet
See [INSTRUCTIONS_SHAPENET.md](INSTRUCTIONS_SHAPENET.md) for more instructions.

## License
The Mesh R-CNN codebase is released under [BSD-3-Clause License](LICENSE)

[1]: https://arxiv.org/abs/1912.08804
[gg]: https://github.com/gkioxari
[jj]: https://github.com/jcjohnson
[d2]: https://github.com/facebookresearch/detectron2
[py3d]: https://github.com/facebookresearch/pytorch3d
