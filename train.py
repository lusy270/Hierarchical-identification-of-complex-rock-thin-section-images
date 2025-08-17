import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model_utils import HierarchicalDataset, AttentionFusionHierarchicalSwinCNN
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    fine_preds, fine_labels = [], []
    coarse_preds, coarse_labels = [], []
    for images, coarse_y, fine_y in dataloader:
        images, coarse_y, fine_y = images.to(device), coarse_y.to(device), fine_y.to(device)
        coarse_out, fine_out = model(images)
        loss = loss_fn(coarse_out, coarse_y) + loss_fn(fine_out, fine_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        fine_preds.append(fine_out.argmax(1).cpu())
        fine_labels.append(fine_y.cpu())
        coarse_preds.append(coarse_out.argmax(1).cpu())
        coarse_labels.append(coarse_y.cpu())
    fine_preds = torch.cat(fine_preds)
    fine_labels = torch.cat(fine_labels)
    coarse_preds = torch.cat(coarse_preds)
    coarse_labels = torch.cat(coarse_labels)
    avg_loss = total_loss / len(dataloader)
    fine_acc = accuracy_score(fine_labels, fine_preds)
    fine_prec = precision_score(fine_labels, fine_preds, average='macro', zero_division=0)
    fine_rec = recall_score(fine_labels, fine_preds, average='macro', zero_division=0)
    fine_f1 = f1_score(fine_labels, fine_preds, average='macro', zero_division=0)
    coarse_acc = accuracy_score(coarse_labels, coarse_preds)
    return avg_loss, fine_acc, fine_prec, fine_rec, fine_f1, coarse_acc
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    fine_preds, fine_labels = [], []
    coarse_preds, coarse_labels = [], []
    with torch.no_grad():
        for images, coarse_y, fine_y in dataloader:
            images, coarse_y, fine_y = images.to(device), coarse_y.to(device), fine_y.to(device)
            coarse_out, fine_out = model(images)
            loss = loss_fn(coarse_out, coarse_y) + loss_fn(fine_out, fine_y)
            total_loss += loss.item()
            fine_preds.append(fine_out.argmax(1).cpu())
            fine_labels.append(fine_y.cpu())
            coarse_preds.append(coarse_out.argmax(1).cpu())
            coarse_labels.append(coarse_y.cpu())
    fine_preds = torch.cat(fine_preds)
    fine_labels = torch.cat(fine_labels)
    coarse_preds = torch.cat(coarse_preds)
    coarse_labels = torch.cat(coarse_labels)
    avg_loss = total_loss / len(dataloader)
    fine_acc = accuracy_score(fine_labels, fine_preds)
    fine_prec = precision_score(fine_labels, fine_preds, average='macro', zero_division=0)
    fine_rec = recall_score(fine_labels, fine_preds, average='macro', zero_division=0)
    fine_f1 = f1_score(fine_labels, fine_preds, average='macro', zero_division=0)
    coarse_acc = accuracy_score(coarse_labels, coarse_preds)
    return avg_loss, fine_acc, fine_prec, fine_rec, fine_f1, coarse_acc
def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0))
        ]),
        transforms.AutoAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = HierarchicalDataset("SplitDataset/train", transform=train_transform)
    test_dataset = HierarchicalDataset("SplitDataset/test", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=60, num_workers=8)
    model = AttentionFusionHierarchicalSwinCNN(num_fine=47).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    num_epochs = 300
    for epoch in range(num_epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_cacc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        test_loss, test_acc, test_prec, test_rec, test_f1, test_cacc = evaluate(
            model, test_loader, loss_fn, device
        )
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("Precision/train", train_prec, epoch)
        writer.add_scalar("Precision/test", test_prec, epoch)
        writer.add_scalar("Recall/train", train_rec, epoch)
        writer.add_scalar("Recall/test", test_rec, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("F1/test", test_f1, epoch)
        writer.add_scalar("CoarseAccuracy/train", train_cacc, epoch)
        writer.add_scalar("CoarseAccuracy/test", test_cacc, epoch)
        scheduler.step()
    torch.save(model.state_dict(), "model_final.pt")
    writer.close()
if __name__ == '__main__':
    main()
