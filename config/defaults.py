from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.PROMPT_NUM = 4 # 1 2 4 8 16 32
_C.MODEL.I2P_LOSS = 1.0 # 1.0
_C.MODEL.T2P_LOSS = 1.0 # 1.0
_C.MODEL.I2TCE_LOSS = 1.0 # 1.0
_C.MODEL.MMM_LOSS_MARGIN = [1.0,1.0,1.0] # 1.0
_C.MODEL.LOSS_TYPE = ['ID_TRIPLE'] # 'ID_TRIPLE' '3M'
_C.MODEL.CM = CN()
_C.MODEL.ADAPTER = 'layer_adapter'
_C.MODEL.MLP_ADAPTER = True
_C.MODEL.MLP_ADP_MID_DIM = 256
_C.MODEL.MLP_ADP_SCALE = 0.1
_C.MODEL.LEARNABLE_SCALAR = False

_C.MODEL.ATT_ADAPTER = False
_C.MODEL.ATT_PREFIX = 'lora_qk' # 'lora_qk', 'patt_kv', 'patt_qv', 'patt_qk', 'patt_qkv', 'prefix_kv', 'prefix_qk'
_C.MODEL.ATT_MID_DIM = 64
_C.MODEL.ATT_PREFIX_SCALE = 0.2

_C.MODEL.CM.USE_HARD = False
_C.MODEL.CM.TEMP = 0.05
_C.MODEL.CM.MOMENTUM = 0.1
_C.MODEL.CM.CHANGE_SCALE = 0.1
_C.MODEL.VISUAL_BRANCH = "single" # multi single
_C.MODEL.TEXT_BRANCH = "single" # multi single
_C.MODEL.MULTI_MODAL = 'RNT' 
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False
_C.MODEL.SIE_SCENE = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# Enable Random Erase in data augmentation
_C.INPUT.RANDOM_ERASE = True # True False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SCHEDULER = 'WarmupMultiStepLR' # StepLR CosineLR WarmupMultiStepLR
_C.SOLVER.COS_CYCLES = 0.5
_C.SOLVER.SCHEDULER = 'step' # step cosine cosine100
_C.SOLVER.PROMPT_LR = 0.00035
_C.SOLVER.PROMPT_WEIGHT_DECAY_BIAS = 1e-4
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_LR_INIT = 0.01
_C.SOLVER.LR_MIN = 0.000016


_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.SAVE_CHECKPOINT = False # True False
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
