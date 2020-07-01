# You Only Watch Once (YOWO)

PyTorch implementation of the article "[You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization](https://arxiv.org/pdf/1911.06644.pdf)".

<br/>
<div align="center" style="width:image width px;">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/biking.gif" width=240 alt="biking">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/fencing.gif" width=240 alt="fencing">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/golf_swing.gif" width=240 alt="golf-swing">
</div>

<div align="center" style="width:image width px;"> 
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/catch.gif" width=240 alt="catch">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/brush_hair.gif" width=240 alt="brush-hair">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/pull_up.gif" width=240 alt="pull-up">
</div>
<br/>
<br/>
  
In this work, we present ***YOWO*** (***Y**ou **O**nly **W**atch **O**nce*), a unified CNN architecture for real-time spatiotemporal action localization in video stream. *YOWO* is a single-stage framework, the input is a clip consisting of several successive frames in a video, while the output predicts bounding box positions as well as corresponding class labels in current frame. Afterwards, with specific strategy, these detections can be linked together to generate *Action Tubes* in the whole video.

Since we do not separate human detection and action classification procedures, the whole network can be optimized by a joint loss in an end-to-end framework. We have carried out a series of comparative evaluations on two challenging representative datasets **UCF101-24** and **J-HMDB-21**. Our approach outperforms the other state-of-the-art results while retaining real-time capability, providing 34 frames-per-second on 16-frames input clips and 62 frames-per-second on 8-frames input clips.


## Installation
```bash
git clone https://github.com/wei-tim/YOWO.git
cd YOWO
```

### Datasets

* UCF101-24: Download from [here](https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view?usp=sharing)
* J-HMDB-21: Download from [here](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets)

Modify the paths in ucf24.data and jhmdb21.data under cfg directory accordingly.

Download the dataset annotations from [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0).

### Download backbone pretrained weights

* Darknet-19 weights can be downloaded via:
```bash
wget http://pjreddie.com/media/files/yolo.weights
```

* ResNeXt ve ResNet pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).

***NOTE:*** For JHMDB-21 trainings, HMDB-51 finetuned pretrained models should be used! (e.g. "resnext-101-kinetics-hmdb51_split1.pth").

* For resource efficient 3D CNN architectures (ShuffleNet, ShuffleNetv2, MobileNet, MobileNetv2), pretrained models can be downloaded from [here](https://github.com/okankop/Efficient-3DCNNs).

### Pretrained YOWO models

Pretrained models can be downloaded from [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0).

All materials (annotations and pretrained models) are also available in Baiduyun Disk:
[here](https://pan.baidu.com/s/1yaOYqzcEx96z9gAkOhMnvQ) with password 95mm

## Running the code

* Example training bash scripts are provided in 'run_ucf101-24.sh' and 'run_jhmdb-21.sh'.
* UCF101-24 training:
```bash
python train.py --dataset ucf101-24 \
		--data_cfg cfg/ucf24.data \
		--cfg_file cfg/ucf24.cfg \
		--n_classes 24 \
		--backbone_3d resnext101 \
		--backbone_3d_weights weights/resnext-101-kinetics.pth \
		--backbone_2d darknet \
		--backbone_2d_weights weights/yolo.weights \
```
* J-HMDB-21 training:
```bash
python train.py --dataset jhmdb-21 \
		--data_cfg cfg/jhmdb21.data \
		--cfg_file cfg/jhmdb21.cfg \
		--n_classes 21 \
		--backbone_3d resnext101 \
		--backbone_3d_weights weights/resnext-101-kinetics-hmdb51_split1.pth \
		--freeze_backbone_3d \
		--backbone_2d darknet \
		--backbone_2d_weights weights/yolo.weights \
		--freeze_backbone_2d \
```

## Validating the model

* After each validation, frame detections is recorded under 'jhmdb_detections' or 'ucf_detections'. From [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0), 'groundtruths_jhmdb.zip' and 'groundtruths_jhmdb.zip' should be downloaded and extracted to "evaluation/Object-Detection-Metrics". Then, run the following command to calculate frame_mAP.

```bash
python evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder PATH-TO-GROUNDTRUTHS-FOLDER --detfolder PATH-TO-DETECTIONS-FOLDER

```

* For video_mAP, 'run_video_mAP_ucf.sh' and 'run_video_mAP_jhmdb.sh' bash scripts can be used.


***UPDATEs:*** 
* We have found a bug in our evaluation for calculating frame-mAP on UCF101-24 dataset (video-mAP results are same as before). We have fixed it, but the frame-mAP results for UCF101-24 are lower than before (if LFB are not used).
* We have used freezing both 2D-CNN and 3D-CNN backbones for all models at the trainings of J-HMDB-21 dataset. This improved the performence considerable, especially for models having resource efficient 3D-CNN backbones.
* We have implemented [Long-Term Feature Bank (LFB)](https://arxiv.org/pdf/1812.05038.pdf). Details can be found in the paper. It brings considerable improvement to UCF101-24 dataset and marginal improvement to J-HMDB-21 dataset. YOWO+LBF achieves 87.3% and 75.7% frame_mAP for UCF101-24 and J-HMDB-21 datasets, respectively.

### Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
@InProceedings{kopuklu2019yowo,
title={You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization},
author={K{\"o}p{\"u}kl{\"u}, Okan and Wei, Xiangyu and Rigoll, Gerhard},
journal={arXiv preprint arXiv:1911.06644},
year={2019}
}
```

### Acknowledgements
We thank [Hang Xiao](https://github.com/marvis) for releasing [pytorch_yolo2](https://github.com/marvis/pytorch-yolo2) codebase, which we build our work on top. 
