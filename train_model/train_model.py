import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from logger import setup_logger

DATA_PATH = "/app/mnist"
TRITON_MODEL_DIR = "/app/models/resnet18_mnist"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1000))
EPOCHS = int(os.getenv("EPOCHS", 10))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
logger = setup_logger(log_file_prefix="train_model")

# MNIST 다운로드
def download_mnist(data_path):
    required_files = [
        os.path.join(data_path, 'MNIST', 'raw', 'train-images-idx3-ubyte.gz'),
        os.path.join(data_path, 'MNIST', 'raw', 'train-labels-idx1-ubyte.gz'),
        os.path.join(data_path, 'MNIST', 'raw', 't10k-images-idx3-ubyte.gz'),
        os.path.join(data_path, 'MNIST', 'raw', 't10k-labels-idx1-ubyte.gz')
    ]
    if all(os.path.exists(file) for file in required_files):
        print("MNIST dataset already exists, skipping download.")
    else:
        print("Downloading MNIST dataset...")
        datasets.MNIST(root=data_path, train=True, download=True)

# 데이터 로더
def get_data_loader(data_path, transform):
    trainset = datasets.MNIST(root=data_path, train=True, transform=transform)
    sampled_indices = random.sample(range(len(trainset)), NUM_SAMPLES)
    sampled_subset = Subset(trainset, sampled_indices)
    return DataLoader(sampled_subset, batch_size=BATCH_SIZE, shuffle=True)

# ResNet18 모델 정의
def create_resnet18():
    model = models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# 모델 학습
def train(model, data_loader, criterion, optimizer, device):
    model.to(device)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(data_loader)

        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}\n")

# torchscript 변환 및 저장
def save_torchscript_model(model, model_dir, device):
    model.eval()
    example_input = torch.rand(1, 1, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    
    model_version_dir = os.path.join(model_dir, "1")
    os.makedirs(model_version_dir, exist_ok=True)
    traced_model.save(os.path.join(model_version_dir, "model.pt"))

    logger.info(f"Model saved at {os.path.join(model_version_dir, 'model.pt')}\n")

# Triton config 생성
def create_triton_config(model_dir):
    config_content = f"""
name: "resnet18_mnist"
platform: "pytorch_libtorch"
input [
  {{
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ -1, 1, 224, 224 ]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }}
]
"""
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
        f.write(config_content)

    logger.info(f"Triton config created at {os.path.join(model_dir, 'config.pbtxt')}\n")


def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    download_mnist(DATA_PATH)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_loader = get_data_loader(DATA_PATH, transform)

    model = create_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

    train(model, train_loader, criterion, optimizer, device)
    save_torchscript_model(model, TRITON_MODEL_DIR, device)
    create_triton_config(TRITON_MODEL_DIR)

if __name__ == "__main__":
    main()
