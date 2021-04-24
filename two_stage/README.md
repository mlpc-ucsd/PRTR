# PRTR Two-stage Cascade Transformers

Implementation of the **two stage variant** in [PRTR: Pose Recognition with Cascade Transformers](https://github.com/mlpc-ucsd/PRTR).

## Main Results
### Results on MPII val
| Arch   | Input Size | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
| -------------- | -------------- | -------- | ------------ | --------- | --------- | ------- | -------- | --------- | -------- | ------------ |
| prtr_res50  | 256x256 | 94.577   | 93.088       | 83.109    | 74.079    | 84.733  | 74.732   | 69.367    | 82.865   | 22.602       |
| prtr_res50  | 384x384 | 96.010   | 94.124       | 86.586    | 79.940    | 86.464  | 81.604   | 74.563    | 86.310   | 28.704       |
| prtr_res50  | 512x512 | 96.555   | 95.007       | 88.597    | 83.383    | 88.731  | 84.002   | 79.121    | 88.493   | 34.036       |
| prtr_res101 | 256x256 | 94.816   | 93.461       | 84.728    | 76.853    | 87.000  | 79.730   | 72.768    | 84.975   | 25.517       |
| prtr_res101 | 384x384 | 96.282   | 94.990       | 88.307    | 82.353    | 88.125  | 83.639   | 77.445    | 87.916   | 31.642       |
| prtr_res101 | 512x512 | 96.828   | 95.839       | 90.234    | 84.633    | 89.302  | 85.049   | 80.043    | 89.409   | 37.198       |
| prtr_res152 | 256x256 | 96.146   | 94.480       | 86.108    | 78.515    | 87.658  | 81.826   | 74.634    | 86.313   | 26.885       |
| prtr_res152 | 384x384 | 96.419   | 94.871       | 88.444    | 82.627    | 88.575  | 84.143   | 78.365    | 88.215   | 32.160       |
| prtr_hrnet_w32  | 256x256 | 96.794   | 95.584       | 89.603    | 83.143    | 88.731  | 83.739   | 78.012    | 88.584   | 33.206       |
| prtr_hrnet_w32  | 384x384 | 97.340   | 96.009       | 90.557    | 84.479    | 89.700  | 85.533   | 78.956    | 89.526   | 35.410       |

### Results on COCO val2017 (with DETR bbox)
| Backbone   | Input Size | AP | AP .50 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
| -------------- | -------------- | ------ | --------- | --------- | -------- | -------- | ------ | --------- | --------- | -------- | -------- |
| prtr_res50  | 384x288      | 68.2   | 88.2      | 75.2      | 63.2     | 76.2     | 76.0   | 92.9      | 82.4      | 70.9     | 83.3     |
| prtr_res50  | 512x384      | 71.0   | 89.3      | 78.0      | 66.4     | 78.8     | 78.0   | 93.5      | 84.1      | 73.0     | 85.1     |
| prtr_res101 | 384x288      | 70.1   | 88.8      | 77.6      | 65.7     | 77.4     | 77.5   | 93.6      | 84.1      | 72.8     | 84.2     |
| prtr_res101 | 512x384      | 72.0   | 89.3      | 79.4      | 67.3     | 79.7     | 79.2   | 93.8      | 85.4      | 74.3     | 86.1     |
| prtr_hrnet_w32  | 384x288      | 73.1   | 89.4      | 79.8      | 68.8     | 80.4     | 79.8   | 93.8      | 85.6      | 75.3     | 86.2     |
| prtr_hrnet_w32  | 512x384      | 73.3   | 89.2      | 79.9      | 69.0     | 80.9     | 80.2   | 93.6      | 85.7      | 75.5     | 86.8     |
| deform_prtr_res50  | 384x288   | 70.8   | 88.5      | 77.5      | 66.8     | 78.3     | 78.3   | 93.2      | 84.1      | 73.4     | 85.1     |
| deform_prtr_res101 | 384x288   | 71.0   | 88.7      | 77.8      | 66.9     | 78.4     | 78.4   | 93.3      | 84.3      | 73.6     | 85.3     |

### Results on COCO test-dev2017
| Backbone   | Input Size | # Params | GFlops | AP | AP .50 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
| -------------- | -------------- | ------------ | ---------- | ------ | --------- | --------- | -------- | -------- | ------ | --------- | --------- | -------- | -------- |
| prtr_res50  | 384x288      | 41.5M        | 11.0       | 67.4   | 89.3      | 75.6      | 62.9     | 74.8     | 75.2   | 93.8      | 82.4      | 70.2     | 82.1     |
| prtr_res50  | 512x384      | 41.5M        | 18.8       | 69.6   | 90.2      | 77.4      | 65.1     | 77.0     | 77.0   | 94.4      | 83.9      | 72.0     | 83.9     |
| prtr_res101 | 384x288      | 60.4M        | 19.1       | 68.8   | 89.9      | 76.9      | 64.7     | 75.8     | 76.6   | 94.4      | 83.7      | 71.8     | 83.0     |
| prtr_res101 | 512x384      | 60.4M        | 33.4       | 70.6   | 90.3      | 78.5      | 66.2     | 77.7     | 78.1   | 94.6      | 84.9      | 73.2     | 84.6     |
| prtr_hrnet_w32 | 384x288      | 57.2M        | 21.6       | 71.7   | 90.6      | 79.6      | 67.6     | 78.4     | 78.8   | 94.7      | 85.7      | 74.4     | 84.8     |
| prtr_hrnet_w32 | 512x384      | 57.2M        | 37.8       | 72.1   | 90.4      | 79.6      | 68.1     | 79.0     | 79.4   | 94.7      | 85.8      | 74.8     | 85.7     |
| deform_prtr_res50  | 384x288  | 46.3M        | 29.1       | 69.3   | 89.5      | 76.9      | 65.5     | 76.3     | 77.3   | 94.1      | 83.9      | 72.4     | 83.9     |
| deform_prtr_res101 | 384x288  | 65.2M        | 37.3       | 69.3   | 89.7      | 76.8      | 65.7     | 76.1     | 77.3   | 94.3      | 83.8      | 72.7     | 83.6     |

#### Note
+ `deform_prtr` is PRTR based on [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159), which uses Deformable Attention and multi-level features. In our case, it converges 
with much fewer epochs compared with vanilla Transformer.

## Getting Started
This project is developed on Ubuntu 18.04 with CUDA 10.2.

### Installation
1. Install pytorch >= v1.4.0 following [official instruction](https://pytorch.org/). We recommend using `conda` virtual environment.
2. Clone this repo, and we'll call the directory `two_stage` as `${POSE_ROOT}`.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download pretrained HRNet weight [hrnetv2_w32_imagenet_pretrained.pth](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/EWCD8fPG2btNg_nNyh9TTIEBCILVnc1onNvK2y8geY2YiQ?e=lVjdLj) and put it into `${POSE_ROOT}/models/pytorch/imagenet/`
5. (Optional) If you want to use the `deformable_pose_transformer` model, please build the Deformable Attention module:
   ```
   cd ${POSE_ROOT}/lib/models/ops/ && bash make.sh
   ```

### Model zoo
Please download our pretrained models from [OneDrive](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/xiz102_ucsd_edu/EneM2Rekv-FCsQAyh8eugM8BUrZ5UXeZdqrteap-Hk38lQ?e=r1ZDaF).
```
${POSE_ROOT}
 `-- models
     `-- pytorch
         |-- imagenet
         |   `-- hrnetv2_w32_imagenet_pretrained.pth
         |-- pose_coco
         |   |-- deform_pose_transformer_res101_384x288.pth
         |   |-- deform_pose_transformer_res50_384x288.pth
         |   |-- pose_transformer_hrnet_w32_384x288.pth
         |   |-- pose_transformer_hrnet_w32_512x384.pth
         |   |-- pose_transformer_res101_384x288.pth
         |   |-- pose_transformer_res101_512x384.pth
         |   |-- pose_transformer_res50_384x288.pth
         |   `-- pose_transformer_res50_512x384.pth
         `-- pose_mpii
             |-- pose_transformer_hrnet_w32_256x256.pth
             |-- pose_transformer_hrnet_w32_384x384.pth
             |-- pose_transformer_res101_256x256.pth
             |-- pose_transformer_res101_384x384.pth
             |-- pose_transformer_res101_512x512.pth
             |-- pose_transformer_res152_256x256.pth
             |-- pose_transformer_res152_384x384.pth
             |-- pose_transformer_res50_256x256.pth
             |-- pose_transformer_res50_384x384.pth
             `-- pose_transformer_res50_512x512.pth
```

### Data preparation
MPII and COCO dataset are supported. Please follow the data downloading and processing guide in [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation#data-preparation). After that, please download COCO person detection bboxes generated by DETR from [OneDrive](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/xiz102_ucsd_edu/EryYO06XFSFAulWksf4NqvcBG2gZM0pJRJOLhaMKwwqAZw?e=EAyGO0) and put them in `${POSE_ROOT}/data/coco/person_detection_results`.
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_test-dev2017_detr_detections.json
        |   `-- COCO_val2017_detr_detections.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
### Visualization
We provide a notebook file [visualization.ipynb](./visualization.ipynb) which helps to visualize the outputs of the model. `pose_transformer_hrnet_w32_384x288.pth` and COCO dataset are needed for it to work properly.

### Training and Testing
Testing on MPII dataset
```
python tools/test.py \
    --cfg experiments/mpii/transformer/w32_256x256_adamw_lr1e-4.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_transformer_w32_256x256.pth
```

Training on MPII dataset
```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adamw_lr1e-4.yaml
```

Testing on COCO val2017 dataset
```
python tools/test.py \
    --cfg experiments/coco/transformer/w32_384x288_adamw_lr1e-4.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_transformer_w32_384x288.pth \
    TEST.USE_GT_BBOX False
```

Training on COCO train2017 dataset
```
python tools/train.py \
    --cfg experiments/coco/transformer/w32_384x288_adamw_lr1e-4.yaml
```

Tracing model stats (Flops, number of params, activations)
```
python tools/trace.py \
    --cfg experiments/coco/transformer/w32_384x288_adamw_lr1e-4.yaml
```

### Train logs
All the training logs can be downloaded from [OneDrive](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/xiz102_ucsd_edu/Ei6QJPBUvrVNsO0vxbpn9AsBOLOhSjVHIPn9zybXAlJUKw?e=2u8S32). Log for `model_name.pth` is named as `model_name.log`.
