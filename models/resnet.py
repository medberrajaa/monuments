import torch
import torchvision.transforms as transforms
import torchvision.models as models
from models.ResidualNetwork import RMN
from PIL import Image

class ResNetModel:
    def __init__(self):
        # Charger un modèle ResNet50 "vide"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RMN().to(self.device)
        # Charger vos poids entraînés
        self.model.load_state_dict(torch.load("weights/ResidualNet.pth", map_location=self.device))
        self.model.eval()

        # Mapping d’indices -> classes (exemple)
        self.class_names = ['Bab El-khamis', 'Bab Mansour', 'Bab berdaine', 'Bab chellah', 'Grande mosquee de Meknes', 'Heri es souani', 'Koutoubia', 'Medersa Attarine', 'Menara', 'Mosque hassa 2', 'Musee Mohammed VI', 'Musee Nejjarin', 'Oualili', 'Palais El Badi', 'Palais Royal de Fes', 'Porte Bab Boujloud', 'Tannerie Chouara', 'Tombeaux saadiens', 'Tour Hassan']
        # Transformations standard
        self.transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image_path):
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        # Transformer l'image
        input_tensor = self.transform(image).unsqueeze(0)  # shape (1, 3, 224, 224)

        # Désactiver le calcul de gradients
        with torch.no_grad():
            outputs = self.model(input_tensor)  # shape (1, nb_classes)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_idx = predicted_idx.item()

        predicted_class = self.class_names[predicted_idx]
        
        # Pour obtenir la probabilité (softmax)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0, predicted_idx].item()

        return f"{predicted_class} (confiance : {confidence:.2f}) via ResNet"
