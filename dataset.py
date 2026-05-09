import os
import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, df, mode="train"):
        self.df = df
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ingredients = self.df.loc[idx, "ingredients"]
        ingredients_arr =  ingredients.split(";")
        if self.mode == "train":
            random.shuffle(ingredients_arr)
        ingredients = ", ".join(ingredients_arr)

        total_mass = torch.tensor(float(self.df.loc[idx, "total_mass"]), dtype=torch.float32)
        total_calories = torch.tensor(float(self.df.loc[idx, "total_calories"]), dtype=torch.float32)
        label = torch.tensor(self.df.loc[idx, "total_calories"] / self.df.loc[idx, "total_mass"], dtype=torch.float32)

        img_path = self.df.loc[idx, "dish_id"]
        try:
            image = Image.open(f"data/images/{img_path}/rgb.jpg").convert('RGB')
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:],
                                           self.image_cfg.input_size[0])).to(
                                               torch.float32)

        image = self.transforms(image=np.array(image))["image"]
        return {"label": label, "image": image, "total_mass": total_mass, "total_calories": total_calories, "ingredients": ingredients} 



def collate_fn(batch, tokenizer):
    ingredients = [item["ingredients"] for item in batch]
    total_mass = [item["total_mass"] for item in batch]
    total_calories = [item["total_calories"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]

    tokenized_input = tokenizer(ingredients,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "label": torch.tensor(labels),
        "image": images,
        "total_mass": torch.tensor(total_mass),
        "total_calories": torch.tensor(total_calories),
        "input_ids": tokenized_input["input_ids"].squeeze(0),
        "attention_mask": tokenized_input["attention_mask"].squeeze(0)
    }

def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.Affine(scale=(0.95, 1.0),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.1,
                              p=0.3),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )

    return transforms
    