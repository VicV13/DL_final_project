import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial
import random
from dataset import MultimodalDataset, collate_fn, get_transforms
import os
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_encoder = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_encoder.num_features, config.HIDDEN_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM*2 + 1, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_encoder(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = torch.cat([text_emb, image_emb, mass.unsqueeze(1)], dim=1)

        return self.classifier(fused_emb)
    
def train(config, device, train_df, test_df):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_encoder,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_encoder,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": config.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": config.IMAGE_LR},
        {"params": model.text_proj.parameters(), "lr": config.CLASSIFIER_LR},
        {"params": model.image_proj.parameters(), "lr": config.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": config.CLASSIFIER_LR},
    ]) 

    criterion = nn.L1Loss() # MAE
    

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    # инициализируем метрику
    train_dataset = MultimodalDataset(config, transforms, train_df)
    val_dataset = MultimodalDataset(config, val_transforms, test_df)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    metric_train = torchmetrics.MeanAbsoluteError().to(device)
    metric_val = torchmetrics.MeanAbsoluteError().to(device)
    best_val = 0.0

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(
                batch['input_ids'].to(device), 
                batch['attention_mask'].to(device),
                batch['image'].to(device), 
                batch['total_mass'].to(device)
            ).view(-1)
            metric_train.update(preds, batch['label'].to(device))
            loss = criterion(preds, batch['label'].to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Валидация
        train_mae = metric_train.compute()
        val_mae = validate(model, val_loader, device, metric_val)
        metric_val.reset()
        metric_train.reset()

        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {total_loss/len(train_loader):.4f} | Train F1: {train_mae :.4f}| Val F1: {val_mae :.4f}"
        )

        if val_mae > best_val:
            print(f"New best model, epoch: {epoch}")
            best_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)

def validate(model, val_loader, device, metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            preds = model(
                    batch['input_ids'].to(device), 
                    batch['attention_mask'].to(device),
                    batch['image'].to(device), 
                    batch['total_mass'].to(device)
                ).view(-1)
            metric.update(preds, batch['label'].to(device))

    return metric.compute()