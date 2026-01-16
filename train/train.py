import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import RSFHModel
from training.utils import HierarchicalDataset
import torchvision.transforms as transforms
import yaml


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_fine_preds, all_fine_labels = [], []
    all_coarse_preds, all_coarse_labels = [], []

    for x, cy, fy in dataloader:
        x, cy, fy = x.to(device), cy.to(device), fy.to(device)
        c_out, f_out = model(x)
        loss = loss_fn(c_out, cy) + loss_fn(f_out, fy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_fine_preds.append(f_out.argmax(1).cpu())
        all_fine_labels.append(fy.cpu())
        all_coarse_preds.append(c_out.argmax(1).cpu())
        all_coarse_labels.append(cy.cpu())

    return total_loss / len(dataloader), all_fine_preds, all_fine_labels, all_coarse_preds, all_coarse_labels


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_fine_preds, all_fine_labels = [], []
    all_coarse_preds, all_coarse_labels = [], []

    with torch.no_grad():
        for x, cy, fy in dataloader:
            x, cy, fy = x.to(device), cy.to(device), fy.to(device)
            c_out, f_out = model(x)
            loss = loss_fn(c_out, cy) + loss_fn(f_out, fy)

            total_loss += loss.item()
            all_fine_preds.append(f_out.argmax(1).cpu())
            all_fine_labels.append(fy.cpu())
            all_coarse_preds.append(c_out.argmax(1).cpu())
            all_coarse_labels.append(cy.cpu())

    return total_loss / len(dataloader), all_fine_preds, all_fine_labels, all_coarse_preds, all_coarse_labels


def main():
    with open('training/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.RandomResizedCrop(config['data']['image_size'], scale=(0.5, 1.0))
        ]),
        transforms.AutoAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = HierarchicalDataset(config['data']['train_path'], transform=train_transform)
    test_dataset = HierarchicalDataset(config['data']['test_path'], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'])

    model = RSFHModel(num_fine=config['model']['num_fine_classes'], num_coarse=config['model']['num_coarse_classes'], dropout_prob=config['model']['dropout_prob']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_decay_steps'], gamma=config['training']['lr_decay_gamma'])
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(config['logging']['log_dir'])

    for epoch in range(config['training']['epochs']):
        train_loss, *train_preds = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, *test_preds = evaluate(model, test_loader, loss_fn, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

    torch.save(model.state_dict(), f"{config['logging']['save_dir']}/model_final.pt")
    writer.close()


if __name__ == '__main__':
    main()