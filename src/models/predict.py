import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from src.models.model import get_model
from src.config.config import Config


config = Config()

# transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


_CACHED_CLASSES = [
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrheic Keratoses"
]
_CACHED_MODEL = None


def get_classes():
    return _CACHED_CLASSES


def load_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = config.get("paths", "model_path")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    _CACHED_MODEL = model
    return _CACHED_MODEL




def predict(image_path):
    model = load_model()
    classes = get_classes()   # ✅ USE HERE

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        

        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)
        return classes[predicted.item()], confidence.item()  # ✅ RETURN NAME
def predict_multiple(image_paths):
    model = load_model()
    classes = get_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, 1)

        label = classes[predicted.item()]
        conf = confidence.item()


        if label not in results:
            results[label] = []

        results[label].append(conf)

    # Voting logic
    final_label = None
    max_score = 0

    for label, confs in results.items():
        avg_conf = sum(confs) / len(confs)

        if avg_conf > max_score:
            max_score = avg_conf
            final_label = label

    return final_label, max_score