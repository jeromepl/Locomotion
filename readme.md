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
