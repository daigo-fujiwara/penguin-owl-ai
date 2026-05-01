import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# --- 1. 画像の前処理（AIが読みやすいサイズに加工） ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

# --- 2. データの読み込み ---
data_dir = 'dataset' # あなたが作ったフォルダ名
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True)
              for x in ['train', 'val']}

# --- 3. 学習済みモデルのロードとカスタマイズ ---
# すでに「1000種類の物体」を知っている賢いモデル(ResNet18)を借りる
model = models.resnet18(pretrained=True)

# 出力部分（最後の出口）を「ペンギンかフクロウか」の2択に作り変える
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- 4. 学習のルール設定 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# --- 5. 学習開始（5周まわす） ---
for epoch in range(5):
    model.train()
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/5 completed.')

# --- 6. 完成した「脳みそ」を保存 ---
torch.save(model.state_dict(), 'penguin_owl_model.pth')
print("学習完了！モデルを保存しました。")
