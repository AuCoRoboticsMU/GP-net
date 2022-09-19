We present the Grasp Proposal Network (GP-net), a Convolutional Neural Network model which can 
generate 6-DOF grasps for mobile manipulators. To train GP-net, we synthetically generate a 
dataset containing depth-images and ground-truth grasp information for more than 1400 objects. 
We use the EGAD! grasping benchmark to evaluate GP-net against two commonly used algorithms, 
the Volumetric Grasping Network (VGN) and the Grasp Pose Detection package (GPD), on a 
PAL TIAGo mobile manipulator in real-world experiments. GP-net achieves grasp success rates 
of 82.2% compared to 57.8% for VGN and 63.3% with GPD. In contrast to the 
state-of-the-art methods in robotic grasping, GP-net can be used out-of-the-box for grasping 
objects with mobile manipulators without limiting the workspace, requiring table segmentation 
or needing additional hardware like high-end GPUs.

This work is currently under review for the International Conference on Robotics and Automation (ICRA) 2023.

The data for training, dataset generation and pre-trained models are available 
on [zenodo](https://zenodo.org/record/7092009#.YyghmtXMJl8).
If you want to use the ROS package on your robot without any model training, 
use [gpnet-ros](https://github.com/AuCoRoboticsMU/gpnet-ros). If you want to train a new model, 
use [the GP-net repository](https://github.com/AuCoRoboticsMU/GP-net). If you want to train a model for an
alternative gripper, use [this code based on DexNet](https://github.com/AuCoRoboticsMU/gpnet-data) to
generate a new dataset

------

If you use our code, please cite

A. Konrad, J. McDonald and R. Villing, "Proposing Grasps for Mobile Manipulators," in review.

along with

J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea,
and K. Goldberg, “Dex-net 2.0: Deep learning to plan robust grasps with
synthetic point clouds and analytic grasp metrics,” in Robotics: Science
and Systems (RSS), 2017.

and

M. Breyer, J. J. Chung, L. Ott, S. Roland, and N. Juan, “Volumetric
grasping network: Real-time 6 dof grasp detection in clutter,” in Conference on Robot Learning, 2020

### Acknowledgements

This publication has emanated from research supported in part by Grants from Science Foundation Ireland under 
Grant numbers 18/CRT/6049 and 16/RI/3399.
The opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do 
not necessarily reflect the views of the Science Foundation Ireland.

### Contact

If you're having questions about any of our projects, please contact [Anna Konrad](mailto:anna.konrad.2020@mumail.ie),
[Prof. John McDonald](mailto:john.mcdonald@mu.ie) or [Dr. Rudi Villing](mailto:rudi.villing@mu.ie).