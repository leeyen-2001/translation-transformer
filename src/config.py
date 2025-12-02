from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
LOGS_DIR = BASE_DIR / 'logs'
NLTK_DATA_DIR = BASE_DIR / 'data' / 'nltk_data'

# 模型参数
DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCONDER_LAYERS = 2
NUM_DECODER_LAYERS = 2

# 训练参数
BATCH_SIZE = 128
SEQ_LEN = 30
LEARNING_RATE = 1e-3
EPOCHS = 30