import cv2
import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader 
import multiprocessing

def main():
    zip_file = "PneumoProcImg/pneumonia.zip"
    ext = "PneumoProcImg/chest_xray"

    if not os.path.exists(ext):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(ext)
    else:
        print("Já extraído!")        

    def process_img(input_dir, output_dir, target_size = (224, 224)):
        os.makedirs(output_dir, exist_ok=True)
        for folder in os.listdir(input_dir):
            class_path = os.path.join(input_dir, folder)
            output_class_path = os.path.join(output_dir, folder)
            os.makedirs(output_class_path, exist_ok=True)
            
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_res = cv2.resize(img, target_size)
                        cv2.imwrite(os.path.join(output_class_path, filename), img_res)

    input_dirs = ["PneumoProcImg/chest_xray1/train", "PneumoProcImg/chest_xray1/test"]
    output_dirs = ["processed_data/train", "processed_data/test"]

    # for input_dir, output_dir in zip(input_dirs, output_dirs):
    #     process_img(input_dir, output_dir)
    # print("Imagens processadas e salvas")

    def remove_file(directory):
        valid = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.lower().endswith(valid) or file == ".DS_Store":
                    file_path = os.path.join(root, file)
                    print(f"Removendo arquivo inválido: {file_path}")
                    os.remove(file_path)

    train_dir = "processed_data/train"
    test_dir = "processed_data/test"

    remove_file(train_dir)
    remove_file(test_dir)

    data_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.ImageFolder(root = train_dir, transform = data_trans)
    test_dataset = datasets.ImageFolder(root = test_dir, transform = data_trans)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True, num_workers=4)

    classes = train_dataset.classes
    print(f"Classes: {classes}")
    print(f"Número de imagens no treino: {len(train_dataset)}")
    print(f"Número de imagens no teste: {len(test_dataset)}")

    
    for i, l in train_loader:
        print(f"Imagem batch: {i.shape}, Labels: {l}")
        break  


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    def train_mod(model, train_loader, test_loader, criterion, optimizer, epochs = 10):
        for ep in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, l in train_loader:
                print(f"Processando batch {ep+1}, lote {i.shape}")  
                i, l = i.to(device), l.to(device)

                optimizer.zero_grad()

                outputs = model(i)
                loss = criterion(outputs, l)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                total += l.size(0)
                correct += (pred == l).sum().item()

            acc = 100 * correct / total
            print(f"Epoch: {ep + 1} / {epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {acc}%")

            evaluate_model(model, test_loader)

    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, l in test_loader:
                i, l = i.to(device), l.to(device)
                outputs = model(i)
                _, pred = torch.max(outputs.data, 1)
                total += l.size(0)
                correct += (pred == l).sum().item()

        acc = 100 * correct / total
        print(f"Accuracy nos testes: {acc}%")

    train_mod(model, train_loader, test_loader, criterion, optimizer)

    torch.save(model.state_dict(), 'modelo_resnet18.pth')

if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn', force = True)
    main()
