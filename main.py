import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


def main():
    # choix du GPU ou du CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # chargement de poids du modele Eff...
    weights = EfficientNet_V2_S_Weights.DEFAULT
    # initialisation du model (reseau)
    net = efficientnet_v2_s(weights=weights)

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(root='birds/Train', transform=transform)
    test_data = torchvision.datasets.ImageFolder(root='birds/Test', transform=transform)

    # remplacement de la dernière couche du réseau de neuronnes pour 
    # une classification en classes
    classes = tuple(train_data.classes)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, len(classes))
    net = net.to(device)

    # split les données en batch
    train_loader = torch.utils.data.DataLoader(
        batch_size=32,
        num_workers=4,
        dataset=train_data,
        shuffle=True,
        persistent_workers=True,
        pin_memory=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-2)
    
    for epoch in range(5):
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            print(f'epoch {epoch + 1}, iteration {i + 1}, loss: {loss.item():.3f}')

if __name__ == '__main__':
    main()