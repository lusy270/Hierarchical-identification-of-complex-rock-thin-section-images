import torch
import torchvision.transforms as transforms
from PIL import Image
from models import RSFHModel
import yaml


def predict_single_image(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        coarse_out, fine_out = model(image)
        coarse_pred = coarse_out.argmax(1).item()
        fine_pred = fine_out.argmax(1).item()

    return coarse_pred, fine_pred


def main():
    with open('training/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = RSFHModel(num_fine=config['model']['num_fine_classes'], num_coarse=config['model']['num_coarse_classes'], dropout_prob=config['model']['dropout_prob']).to(device)
    model.load_state_dict(torch.load(f"{config['logging']['save_dir']}/model_final.pt"))
    model.eval()

    image_path = "path/to/image.jpg"
    coarse_pred, fine_pred = predict_single_image(image_path, model, val_transform, device)

    print(f"Coarse prediction: {coarse_pred}")
    print(f"Fine prediction: {fine_pred}")


if __name__ == '__main__':
    main()