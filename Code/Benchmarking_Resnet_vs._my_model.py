"Danial Yntykbay, LSP: 2330236"

"""
Semantic Segmentation Pipeline using FiftyOne and PyTorch

This script handles:
- Dataset loading and preprocessing
- Custom dataset class definition
- Model implementation (custom U-Net and DeepLabV3)
- Training loop
- Evaluation with metrics
"""

import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import torch.nn as nn
import os
from fiftyone import ViewField as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import datetime
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report
import torchvision.models as models

# Configuration
CLASSES = ["Person", "Dog", "Car"]
CLASS_TO_ID = {"Person": 1, "Dog": 2, "Car": 3}  # Class to ID mapping
RESIZE_SHAPE = (256, 256)
BATCH_SIZE = 16
N_EPOCHS = 8
TRAIN_SIZE = 6000
TEST_SIZE = 100

# Dataset Preparation ---------------------------------------------------------
print("Available datasets:", fo.list_datasets())
classes  = ["Person", "Dog", "Car"]
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["segmentations"],
    classes=classes,
    max_samples=6100,
    shuffle=True,
    only_matching=True
)

# Load Open Images v7 dataset
if "open-images-v7-train-6100" in fo.list_datasets():
    dataset = fo.load_dataset("open-images-v7-train-6100")
    print(f"Dataset loaded with {len(dataset)} samples")
else:
    print("Dataset not found. Download required.")
    # Note: Add dataset download logic here if needed

# Process masks and create multi-class segmentation maps
print("\nProcessing masks...")
for sample in dataset:
    if "ground_truth" in sample:
        # Filter detections for target classes
        filtered_detections = [
            det for det in sample.ground_truth.detections
            if det.label in CLASSES
        ]

        # Create blank mask
        img = Image.open(sample.filepath)
        width, height = img.size
        mask = np.zeros((height, width), dtype=np.uint8)

        # Combine class masks
        for det in filtered_detections:
            if hasattr(det, 'mask'):
                det_mask = det.mask
                # Resize mask if necessary
                if det_mask.shape != (height, width):
                    det_mask = cv2.resize(
                        det_mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
                mask[det_mask > 0] = CLASS_TO_ID[det.label]

        # Save processed mask
        mask_path = f"{os.path.splitext(sample.filepath)[0]}_mask.png"
        Image.fromarray(mask).save(mask_path)
        sample["processed_mask"] = mask_path
        sample.save()


# Dataset Class ----------------------------------------------------------------
class SegmentationDataset(Dataset):
    """Custom dataset for segmentation tasks

    Args:
        samples (list): List of FiftyOne samples
        resize_shape (tuple): Target size for resizing (H, W)
    """

    def __init__(self, samples, resize_shape=RESIZE_SHAPE):
        self.samples = samples
        self.resize_shape = resize_shape

        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Mask transformations
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.resize_shape, interpolation=Image.NEAREST),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample.filepath).convert("RGB")
        image = self.image_transform(image)

        # Load and transform mask
        mask = Image.open(sample["processed_mask"])
        mask = self.mask_transform(mask).squeeze(0)

        return image, mask.long()


# Create and split dataset
full_dataset = SegmentationDataset(list(dataset))
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(
    full_dataset, [TRAIN_SIZE, TEST_SIZE], generator=generator
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Save datasets
torch.save(train_dataset, 'train_dataset.pth')
torch.save(test_dataset, 'test_dataset.pth')


# Model Architecture -----------------------------------------------------------
class Net(nn.Module):
    """Custom U-Net style architecture with dropout regularization"""

    def __init__(self, dropout_prob=0.1):
        super().__init__()
        # Encoder
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout2d(dropout_prob)

        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)  # 256→128

        # Middle
        self.conv2_1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(8)
        self.dropout2 = nn.Dropout2d(dropout_prob)

        self.conv2_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(8)

        # Decoder
        self.up1 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)  # 128→256
        self.upconv = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn_up = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(dropout_prob)

        self.up2 = nn.Conv2d(16, 4, kernel_size=1)  # Final output channels

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # Middle
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn2_2(self.conv2_2(x)))

        # Decoder
        x = F.relu(self.up1(x))
        x = F.relu(self.bn_up(self.upconv(x)))
        x = self.dropout3(x)
        return self.up2(x)


# Training Utilities -----------------------------------------------------------
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    """Basic training loop with progress tracking

    Args:
        n_epochs (int): Number of training epochs
        optimizer: Optimization algorithm
        model: Model to train
        loss_fn: Loss function
        train_loader: Training data loader
    """
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"{datetime.datetime.now()} Epoch {epoch}, "
              f"Loss: {epoch_loss / len(train_loader):.4f}")


# Initialize Model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# Train Custom Model
print("\nTraining Custom Model...")
training_loop(
    n_epochs=N_EPOCHS,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
)


# Evaluation --------------------------------------------------------------------
def evaluate_model(model, loader):
    """Evaluate model performance using classification metrics

    Args:
        model: Trained model
        loader: Data loader for evaluation

    Returns:
        Classification report string
    """
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_labels.append(labels.numpy())
            pred_labels.append(preds.numpy())

    # Flatten all predictions
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)

    return classification_report(
        true_labels, pred_labels,
        target_names=["Background", "Person", "Dog", "Car"]
    )


print("\nCustom Model Evaluation:")
print(evaluate_model(model, test_loader))

# Benchmark with ResNet --------------------------------------------------------
print("\nBenchmarking with DeepLabV3-ResNet101...")

# Modify pretrained model
model_resnet = models.segmentation.deeplabv3_resnet101(pretrained=True)
model_resnet.classifier[4] = nn.Conv2d(256, 4, kernel_size=1)
model_resnet.aux_classifier[4] = nn.Conv2d(256, 4, kernel_size=1)

# Train ResNet model
training_loop(
    n_epochs=N_EPOCHS,
    optimizer=optimizer,
    model=model_resnet,
    loss_fn=loss_fn,
    train_loader=train_loader,
)

# Evaluate ResNet model
print("\nResNet Model Evaluation:")
print(evaluate_model(model_resnet, test_loader))