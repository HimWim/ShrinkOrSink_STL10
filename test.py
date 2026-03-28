import argparse
import torch
from torchvision import datasets, transforms
from model import get_model


def get_loader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    dataset = datasets.STL10(
        root=data_path,
        split='test',
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return correct, total, accuracy


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)

    loader = get_loader(args.data_path, args.batch_size)

    correct, total, acc = evaluate(model, loader, device)

    print("\n===== RESULTS =====")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)