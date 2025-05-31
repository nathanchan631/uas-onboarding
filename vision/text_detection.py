import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

def get_finetuned_resnet(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Define the same transform as in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_text(image_cv, model_path='resnet18_finetuned.pth', topk=3):
    # Load class mapping from CSV
    csv_path = 'model/text_labels.csv'
    df = pd.read_csv(csv_path)
    classes = sorted(df['label'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_finetuned_resnet(num_classes=len(class_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    model.to(device)

    image = Image.fromarray(image_cv)
    image = image.convert('L')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=topk, dim=1)
        top_probs = top_probs.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()
        top_labels = [idx_to_class[idx] for idx in top_indices]
    return list(zip(top_labels, top_probs))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>") # Ex: python3 inference.py img.png
        sys.exit(1)
    image_path = sys.argv[1]
    top_preds = predict_text(image_path)
    print("Top 3 predictions:")
    for i, (label, prob) in enumerate(top_preds, 1):
        print(f"{i}. {label}: {prob:.4f}") 