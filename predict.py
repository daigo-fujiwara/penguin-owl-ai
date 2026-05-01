import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# 1. AIのモデル構造を定義（ResNet18という型番を使います）
def build_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    # 出口を「2択（フクロウ or ペンギン）」に設定
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# 2. 画像をAIが読める形式（224x224サイズなど）に変換する設定
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path, model_path='penguin_owl_model.pth'):
    # クラス名（アルファベット順が基本）
    class_names = ['owl', 'penguin']

    # モデルの準備と学習済みデータの読み込み
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: {model_path} が見つかりません。先に学習(train.py)を行ってください。")
        return

    model.to(device)
    model.eval()

    # 画像の読み込みと判定
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        with torch.no_grad():
            out = model(batch_t)
            prob = torch.nn.functional.softmax(out, dim=1)[0]
            confidence, index = torch.max(prob, 0)
        
        print(f"--- 判定結果 ---")
        print(f"これは... {class_names[index]} です！ (確信度: {confidence.item()*100:.2f}%)")
    except Exception as e:
        print(f"Error: 画像の読み込みに失敗しました。 {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("使い方: python predict.py [画像ファイルのパス]")
