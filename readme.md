
The goal of this project was to attempt to replicate the results from [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html), using Tensorflow, while making simplifications and generalizations in order to reduce the software engineering cost of such an algorithm. In this work, Peng et al. trained humanoid agents in a 3D physics-based environment to replicate human motions recorded in the form of mocap data files, while also achieving a task objective. The results are very impressive, and I highly recommend everyone to watch a few of the results [here](https://www.youtube.com/watch?v=vppFvq2quQ0).

Note that since October 2018, the source code for DeepMimic is now available [here](https://github.com/xbpeng/DeepMimic).

The conclusion for this project was that the many ablations and generalizations that were performed in order to simplify the system's architecture proved to be too significant for the reinforcement learning agent to learn to successfully replicate the reference human motions. However, addition an imitation component to the reward function was shown to significantly help the agent to more easily learn a successful walking motion and to learn motions that are closer to their corresponding reference motion.

This was done as part of the courses ECSE498 and ECSE499 at McGill, as my final project for my Honours Electrical Engineering bachelors degree.

## Setup

Run the following:

```bash
pip install -r requirements.txt
```

## Run

```bash
# For a list of all arguments
python train.py --help

# To run and show the humanoid while training
python train.py --render
```

## Credits

-   PPO code adapted from https://github.com/pat-coady/trpo
-   Gym environment code adapted from [pybullet](https://github.com/bulletphysics/bullet3)
-   [CMU mocap database](http://mocap.cs.cmu.edu/), along with the bvh format conversion found [here](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/cmu-bvh-conversion)
-   Special thanks to @omimo for releasing the awesome .bvh parsing tool that is [PyMO](https://github.com/omimo/PyMO)
