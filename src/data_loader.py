# data_loader.py

import os
import torch
from torchvision import transforms, datasets

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.classes = sorted(os.listdir(data_dir))
        self.image_paths = []
        self.labels = []

        for i, person in enumerate(self.classes):
            person_path = os.path.join(data_dir, person)
            for image in os.listdir(person_path):
                image_path = os.path.join(person_path, image)
                self.image_paths.append(image_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
