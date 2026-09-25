from datetime import datetime

import torch.nn.functional as F
from PIL import Image
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

def main():
    # choix du GPU ou du CPU
    type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())
    
    device = torch.device(type)

    # # chargement de poids du modele Eff...
    # weights = EfficientNet_V2_S_Weights.DEFAULT
    # # initialisation du model (reseau)
    # net = efficientnet_v2_s(weights=weights)

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(root='birds/Train', transform=transform)
    # test_data = torchvision.datasets.ImageFolder(root='birds/Test', transform=transform)

    # # remplacement de la dernière couche du réseau de neuronnes pour 
    # # une classification en classes
    classes = tuple(train_data.classes)
    # in_features = net.classifier[1].in_features
    # net.classifier[1] = nn.Linear(in_features, len(classes))
    # net = net.to(device)

    # # split les données en batch
    # train_loader = torch.utils.data.DataLoader(
    #     batch_size=32,
    #     num_workers=4,
    #     dataset=train_data,
    #     shuffle=True,
    #     persistent_workers=True,
    #     pin_memory=True
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=32,
    #     num_workers=4,
    #     dataset=test_data,
    #     shuffle=True,
    #     persistent_workers=True,
    #     pin_memory=True
    # )

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-2)
    
    # for epoch in range(5):
    #     net.train()
    #     for i, data in enumerate(train_loader, 0):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         optimizer.zero_grad()
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    #         optimizer.step()
    #         print(f'epoch {epoch + 1}, iteration {i + 1}, loss: {loss.item():.3f}')

    # correct = 0
    # incorrect = 0
    # with torch.no_grad():
    #     net.eval()
    #     for data in test_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         correct += (predicted == labels).sum().item()
    #         incorrect += (predicted != labels).sum().item()
    #         print(correct, '/', incorrect + correct)
    #         print('accuracy', (correct / (incorrect + correct)) * 100, '%')

    # torch.save(net.to(type).state_dict(), f'birds_model{datetime.now().isoformat()}.pt')
    net = efficientnet_v2_s()
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, 200)
    net = net.to(device)
    net.load_state_dict(torch.load('birds_model.pt', weights_only=True))

    with torch.no_grad():
        net.eval()
        image = Image.open('images.jpg').convert("RGB")
        x = transform(image).unsqueeze(0).to(device)
        result = net(x)
        # 3. Convert raw logits into class probabilities (0% to 100%)
        probabilities = F.softmax(result, dim=1)
        
        # 4. Extract the top 5 highest probabilities and their corresponding class indices
        top5_probs, top5_classes = torch.topk(probabilities, k=5, dim=1)
        print("--- Top 5 Predictions ---")
    for rank in range(5):
        class_id = top5_classes[0][rank].item()
        confidence = top5_probs[0][rank].item() * 100
        
        # If you have a class names list/tuple (e.g., classes = train_data.classes):
        # class_name = classes[class_id]
        # print(f"Rank {rank + 1}: {class_name} (ID: {class_id}) - {confidence:.2f}%")
        
        print(classes[class_id])
        print(f"Rank {rank + 1}: Class ID {class_id} - {confidence:.2f}%")

if __name__ == '__main__':
    main()