# Data params
SAVE_PATH = 'model/iter_{}.tf'
TEST_ITER = 30000
RELATIVE_PATH = True
KITTIPATH = '/home/fto2bp/stereo/data_scene_flow'
#KITTIPATH = '/mnt/raid_data/fto2bp/stereo'

MAX_DISPARITY = 128

# Network params
BATCH_SIZE = 128
CHANNELS = 3
FILTERS = 64
KERNEL = 3
CONV_LAYERS = 4

# Training hyperparameters
LOSS_WEIGHTS = [0.05, 0.2, 0.5, 0.2, 0.05]
MAX_EPOCHS = 300
EPOCH_ITERS = 50
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
