class Config:
    SEED = 42
    IMG_DIR = "data/images"
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "resnet50"

    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"  
    IMAGE_MODEL_UNFREEZE = "layer.3|layer.4"
    PROJ_DIM = 128
    BATCH_SIZE = 4
    EPOCHS = 15
    IMAGE_LR = 5e-4
    TEXT_LR = 1e-5
    CLASSIFIER_LR = 1e-3

    HIDDEN_DIM = 128
    SAVE_PATH = "data/best_model.pth"