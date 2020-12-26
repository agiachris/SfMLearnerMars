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

Sample Pose Estimation (Run2 Trajectory)
![][run2d] ![][run3d]

[disp1]: https://github.com/agiachris/SfMLearnerMars/tree/main/misc/train_s0_disp.png "Disparity Sample 1"
[img1]: https://github.com/agiachris/SfMLearnerMars/tree/main/misc/train_s0_tgt_img.png "Target Image 1"

[disp2]: https://github.com/agiachris/SfMLearnerMars/tree/main/misc/train_s2_disp.png "Disparity Sample 2"
[img2]: https://github.com/agiachris/SfMLearnerMars/tree/main/misc/train_s2_tgt_img.png "Target Image 2"

[run2d]: https://github.com/agiachris/SfMLearnerMars/tree/main/misc/epo0_run2_umeyama_traj_overlap.png "Run2 BEV Trajectory"
[run3d]: https://github.com/agiachris/SfMLearnerMars/tree/main/misc/epo0_run2_umeyama_3Dtraj_overlap.png "Run2 3D Trajectory"

### Requirements
- OpenCV (4.5.0)
- PyTorch (1.4.0)
- MatPlotLib (3.3.3)
- NumPy (1.19.4)

### What's been done
- Stuff


### Training

### Evaluation

## References
1. Lamarre, O., Limoyo, O., Marić, F., & Kelly, J. (2020). The Canadian Planetary Emulation
Terrain Energy-Aware Rover Navigation Dataset. The International Journal of Robotics
Research, 39(6), 641-650. doi:10.1177/0278364920908922
2. Zhou, T., Brown, M., Snavely, N., & Lowe, D. G. (2017). Unsupervised Learning of Depth and
Ego-Motion from Video. 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). doi:10.1109/cvpr.2017.700
