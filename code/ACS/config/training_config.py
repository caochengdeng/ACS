import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingConfig:
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    NUM_EPISODES = 5000
    MAX_STEP = 1000000
    GAMMA = 0.96
    ACTOR_LR = 0.001
    CRITIC_LR = 0.001
    SAFETY_LR = 0.001
    VF_COEF = 0.1
    ENT_COEF = 0.01
    MAX_GRAD_NORM = 1.5
    BATCH_SIZE = 256

    LMBDA = 0.95
    EPOCHS = 10
    EPS = 0.2

    ENV_NAME = "highway-fast-v0"

    LOG_DIR = "."
    writer = SummaryWriter(LOG_DIR)

    CAPACITY = 2048

    SEED = 12
    MAX_SIMULATION_STEP = 100


def get_writer():
    return TrainingConfig.writer