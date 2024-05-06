
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from param import parse_args
from data import ImageDataset

class ResNet50(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.model = models.resnet50(weights="IMAGENET1K_V2")
        # 设置所有层为不更新
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.model.fc.requires_grad = True
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        feature = self.model(x)
        out = self.fc(feature)
        return feature, out
    
def train():
    args = parse_args()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_train = ImageDataset(args, mode='train', transform=transform)
    data_test = ImageDataset(args, mode='test', transform=transform)

    device = f'cuda:{args.ss_device}' if args.ss_device >= 0 else 'cpu'
    model = ResNet50(num_classes=50)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    data_train = DataLoader(data_train, batch_size=256, shuffle=True)
    data_test = DataLoader(data_test, batch_size=256, shuffle=False)

    f = open('results/resnet50.txt', 'a')

    best_acc = 0

    for epoch in range(50):
        model.train()
        with tqdm(total=len(data_train), desc=f"Epoch {epoch} [train]") as pbar:
            for filename, images, labels in data_train:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                _, outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(data_test), desc=f"Epoch {epoch} [test]") as pbar:
                for filename, images, labels in data_test:
                    images = images.to(device)
                    labels = labels.to(device)
                    _, outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar.update(1)
                    pbar.set_postfix_str(f"Accuracy: {correct / total:.4f}")
        if correct / total > best_acc:
            best_acc = correct / total
            print(f"save model at epoch {epoch} with accuracy {best_acc:.4f}")
            torch.save(model, "resnet50_best.pth")
            f.write(f"Epoch {epoch}: {best_acc:.4f} [model saved]\n")
        else:
            f.write(f"Epoch {epoch}: {correct / total:.4f}\n")
        f.flush()
    f.close()

if __name__ == '__main__':
    train()