# utils.py

from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    return image

def preprocess_image(image):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image = transform(image)
    return torch.unsqueeze(image, 0)
