import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class SampleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer-1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # Layer-2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Layer-3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Layer-4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)
        # Layer-5
        self.conv5 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        # Layer-6
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.AdaptiveAvgPool2d((1, 1))
        # FC
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{total_epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*correct/total:.2f}")
    return running_loss / total, 100.0 * correct / total

def test_one_epoch(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Test  Epoch {epoch+1}/{total_epochs}")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*correct/total:.2f}")
    return running_loss / total, 100.0 * correct / total

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Dataset setup ---
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    perm = torch.randperm(len(trainset))[:32]
    calibset = Subset(trainset, perm.tolist())

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader  = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    calibloader = DataLoader(calibset, batch_size=32, shuffle=False, num_workers=2)

    # --- Model initialization ---
    model = SampleCnn().to(device)
    print("[x] Using SampleCnn")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # --- Calibration Pass ---
    model.eval()
    with torch.no_grad():
        for images, labels in calibloader:
            images = images.to(device)
            _ = model(images)
            break

    # --- Main Loop ---
    epochs = 2
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, epochs)
        test_loss, test_acc = test_one_epoch(model, testloader, criterion, device, epoch, epochs)

        if (epoch + 1) % 10 == 0:
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

        print(f"\nEpoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n")

    print("Final results reached.")