# SfMLearnerMars
SfMLearner applied to the Canadian Planetary Emulation Terrain Energy-Aware Rover Navigation Dataset (CPET) [1].

This code base is largely built off the [SFMLearner PyTorch](https://github.com/ClementPinard/SfmLearner-Pytorch) 
implementation by Clément Pinard. The original project page of "Unsupervised Learning of Depth and 
Ego-Motion from Video" [2] by [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), 
[Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), 
and [David Lowe](https://www.cs.ubc.ca/~lowe/home.html) can be found [here](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/).

Sample Disparities:

![][img1] ![][disp1]
![][img2] ![][disp2]

[disp1]: https://github.com/agiachris/SfMLearnerMars/blob/main/misc/train_s0_disp.png "Disparity Sample 1"
[img1]: https://github.com/agiachris/SfMLearnerMars/blob/main/misc/train_s0_tgt_img.png "Target Image 1"

[disp2]: https://github.com/agiachris/SfMLearnerMars/blob/main/misc/train_s2_disp.png "Disparity Sample 2"
[img2]: https://github.com/agiachris/SfMLearnerMars/blob/main/misc/train_s2_tgt_img.png "Target Image 2"

[run2d]: https://github.com/agiachris/SfMLearnerMars/blob/main/misc/epo0_run2_umeyama_traj_overlap.png "Run2 BEV Trajectory"
[run3d]: https://github.com/agiachris/SfMLearnerMars/blob/main/misc/epo0_run2_umeyama_3Dtraj_overlap.png "Run2 3D Trajectory"

### About
The goal of this project is to investigate the feasibility of SfMLearner for tracking in
low-textured martian-like environments from monochrome image sequences. The Canadian Planetary
Emulation Terrain Energy-Aware Rover Navigation Dataset provides the necessary data to explore this idea.
On a high-level, here is what's been done:
- Supervised depth pre-training pipeline if you wish to accelerate the joint learning of pose and depth 
by leveraging ground-truth pose 
- Unsupervised learning of motion and depth on the CPET dataset (with option to use pre-trained depth weights)
- Methods for generating, aligning, and plotting (2D & 3D) of absolute trajectories from relative pose estimates during online training
- An independent evaluation pipeline that generates quantitative metrics (ATE) on a specified sequence with pre-trained depth and pose CNN weights

### Requirements
- OpenCV (4.5.0)
- PyTorch (1.4.0)
- MatPlotLib (3.3.3)
- NumPy (1.19.4)

### Data
Training and evaluation require no pre-processing of data. Simply download the "Human-readable (base) data download"
runs of interest from the [CPET webpage](https://starslab.ca/enav-planetary-dataset/?fbclid=IwAR1wZPkyNQ569TCzianx9hzElKHwqqfffV-uvpzMImia2IQqNTGyn4IjBPw).
You'll also need [rover transform](ftp://128.100.201.179/2019-enav-planetary/rover_transforms.txt) and 
[camera intrinsics](ftp://128.100.201.179/2019-enav-planetary/cameras_intrinsics.txt) files. Unpack these into the 
following directory structure:
```
/run1_base_hr
/run2_base_hr
/run3_base_hr
/run4_base_hr
/run5_base_hr
/run6_base_hr
/cameras_intrinsics.txt
/rover_transforms.txt
```

Any missing files (e.g. global-pose-utm for run5) can be manually downloaded from the [dataset drive](https://drive.google.com/drive/folders/1CaMLbStUyySUBSnVNizBqBAsIyFV_Llu).


### Training

#### Supervised Depth Pre-training
There are potentially many factors that inhibit SfMLearner in martian-like environments. For instance, pixel regions 
across images are extremely similar due to the scene's homogenous nature, and operating on monochrome images offer 
lower pixel variance. So, unsupervised learning from scratch may take much longer to converge than expected - or might
not converge. 

After downloading the CPET data, you can train the depth network with the below command. Additional flags can be found 
in [train_depth.py](https://github.com/agiachris/SfMLearnerMars/blob/main/train_depth.py).
```python
python train_depth.py --exp-name <exp-name> --dataset-dir <path/to/data/root>
```

Alternatively, you can download the pre-trained depth network weights (epoch 5) from this 
[link](https://drive.google.com/file/d/1R6mspmyvz_wO7DCmGFCK96AElrXYnSBe/view?usp=sharing).

#### Joint Depth and Pose Learning
Jointly train the pose and depth network with:
```python
python train_joint.py --exp-name <exp-name> --dataset-dir <path/to/data/root> --disp-net <path/to/pre-trained/weights>
```
The --disp-net flag is optional - if left unfilled, the script will default to training the depth and pose network from
scratch in fully unsupervised fashion. The program will save plots of estimated trajectory on the validation sequence
at each epoch. A sample plot n run2_base_hr sequence is given below. Furthermore, quantitative metrics / model 
checkpoints will be saved in the experiment directory.  

Sample Pose Estimation (Run2 Trajectory)
![][run2d]

### Evaluation

The [evaluate_joint.py](https://github.com/agiachris/SfMLearnerMars/blob/main/evaluate_joint.py) script is used to
evaluate the trained models trained models on the test sequence (run6), but it works just fine on any of the training / 
validation sequences too. You can run evaluation on --run-sequence equal to one of 'run1'-'run6' with:
```python
python evaluate_joint.py --exp-name <exp-name> --run-sequence <seq> --dataset-dir <path/to/data/root> --disp-net <path/to/depth/weights> --pose-net <path/to/pose/weights>
```

Sample Pose Estimation (Run2 Trajectory)
![][run3d]


## References
1. Lamarre, O., Limoyo, O., Marić, F., & Kelly, J. (2020). The Canadian Planetary Emulation
Terrain Energy-Aware Rover Navigation Dataset. The International Journal of Robotics
Research, 39(6), 641-650. doi:10.1177/0278364920908922
2. Zhou, T., Brown, M., Snavely, N., & Lowe, D. G. (2017). Unsupervised Learning of Depth and
Ego-Motion from Video. 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). doi:10.1109/cvpr.2017.700
