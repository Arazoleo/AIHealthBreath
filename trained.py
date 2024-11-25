import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import io


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights = None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('modelo_resnet18.pth', weights_only = True))
model.to(device)
model.eval()



data_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])


def predict_img(img):
    img_ten = data_t(img).unsqueeze(0).to(device)  # Converter a imagem para tensor
    with torch.no_grad():
        output = model(img_ten)  # Obter a saída do modelo
    _, pred = torch.max(output, 1)  # Obter a classe com maior probabilidade
    return pred.item()


st.title("AI HealthBreath")
st.write("Envie uma imagem para classificar se a pessoa tem pneumonia ou está saudável.")


upload = st.file_uploader("Enviar imagem", type=["jpg", "jpeg", "png"])


if upload is not None:
    
    img = Image.open(upload)
    st.image(img, caption="Imagem carregada", use_column_width=True)
    
    
    pred_cl = predict_img(img)

    
    classes = ['Normal', 'Pneumonia']
    
    if classes[pred_cl] == 'Normal':
        st.write(f"Relaxe ! O pulmão isento de pneumonia !")
    else:
        st.write(f"Alerta ! O pulmão tem pneumonia !")

