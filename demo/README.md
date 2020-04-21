Run Mesh R-CNN on an input image

```
python demo/demo.py \
--config-file configs/pix3d/meshrcnn_R50_FPN.yaml \
--input /path/to/image \
--output output_demo \
--onlyhighest MODEL.WEIGHTS meshrcnn://meshrcnn_R50.pth
```

The `--onlyhighest` flag will return the highest scoring object prediction. If you remove this flag, all predictions will be returned.

Here are some notes to clarify and guide you how to use the outputs from Mesh R-CNN.

### What does Mesh R-CNN output?
The Mesh R-CNN demo will detect the objects in the image from the Pix3D vocabulary of objects, along with their 2D bounding boxes, 2D instance masks and 3D meshes. For each detected object, Mesh R-CNN returns the 3D shape of an object in the camera coordinate system confined in a 3D box which respects the aspect ratio of the object detected in the image. If you provide the _focal length_ `f` of the camera and the actual depth location `t_z` of the object's center , i.e. how far the center of the object is from the image plane in the Z axis, then Mesh R-CNN will pixel align the predicted 3D object shape with the image *and* the prediction would correspond to the true metric size of the object - its actual scale in the real world!.

### Metric scale
While most images nowadays have access to their focal_length `f` from the image metadata, knowing `t_z` is difficult. We could of course supervise for `t_z` but Pix3D does not contain useful object metric depth. In the Pix3D annotations, the tuple `(f, t_z)` provided does not correspond to the actual camera metadata nor metric depth of the object but is computed subsequently at annotation time by their annotation process and annotation tool and thus is somewhat adhoc. This is the reason we don't tackle the problem of estimating `t_z` (this problem is also called the scene layout prediction problem).

### I don't care about metric scale. I just want to pixel align via rendering.
However if you don't care about metric scale and you only care about pixel aligning the object to the image, that is possible with our demo! **The demo runs with a default focal_length `f=20`** (this is the blender focal length assuming 32mm sensor width and is *not* the true focal_length of the image! We make it up!). The demo also places the object at some arbitrary `t_z > 0.0`, again this is not the true metric depth of the object. Given these choices of `(f, t_z)`, the demo will output an object shape placed at `t_z`.  The metric size of the predicted object from the demo will not correspond to the true size of the object in the world, but it will be a scaled version of it. Now to pixel align the predicted shape with the image, **all you need to do is render the 3D mesh with `f=20`**. _Note that the value 20 is inconsequential. You would be getting the same pixel alignment if `f` was something else, but it's important that the value of `f` you pick when running the demo is also used when rendering!_

Here is an example! When I run the demo on an image from Pix3D (1st image), it recognizes the sofa (2nd image). I get a 3D shape prediction for the sofa which after **I render with blender with focal length `f=20`** I get the final result (3rd image).

![input](https://user-images.githubusercontent.com/4369065/77708628-cda99d00-6f85-11ea-949a-5dad891005ee.jpg)
![segmentation](https://user-images.githubusercontent.com/4369065/77709133-52e18180-6f87-11ea-901a-0706c3d4e3a3.png)
![rendered_output](https://user-images.githubusercontent.com/4369065/77708647-df8b4000-6f85-11ea-8d5f-4ae62ea3bf07.png)
