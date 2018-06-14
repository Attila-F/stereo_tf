# stereo_tf
Tensorflow implementation of Luo et. al.: Efficient Deep Learning for Stereo Matching

To use:

Edit config.py

-- Preprocess data: python stereo.py --mode preprocess

-- Train model: python stereo.py --mode train

-- Test model: python stereo.py --mode test

Unsmoothed after 30000 iterations

2px error:  0.642944731057

3px error:  0.573093536306

4px error:  0.525425039494

5px error:  0.485655262315

7px error:  0.423360325228

10px error:  0.353741120845

20px error:  0.213077628396

30px error:  0.132831055777
