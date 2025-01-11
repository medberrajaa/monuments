import tensorflow as tf
import numpy as np
from PIL import Image

class SimpleCNNModel:
    def __init__(self):
        # On charge le modèle Keras (architecture + poids)
        # Assurez-vous que le chemin correspond à l'emplacement du fichier
        self.model = tf.keras.models.load_model("weights/best_cnn_model.keras")

        # Exemple d’un mapping d’indices -> classes
        # Adaptez ceci en fonction de votre dataset réel
        self.class_names = [
            "Hassan Tower",
            "Koutoubia",
            "Bab Boujloud",
            "Chellah"
        ]

    def predict(self, image_path):
        # Ouvrir l'image avec PIL
        img = Image.open(image_path).convert('RGB')
        # Redimensionner l’image selon l’entrée attendue par votre modèle
        img = img.resize((150, 150))
        
        # Normaliser et convertir en tableau numpy
        img_array = np.array(img) / 255.0  # Normalisation 0-1
        # Ajouter une dimension batch
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # Prédiction
        predictions = self.model.predict(img_array)  # forme (1, nb_classes)
        predicted_index = np.argmax(predictions[0])   # index de la classe la plus probable
        predicted_class_name = self.class_names[predicted_index]

        # Vous pouvez récupérer la probabilité associée pour plus d’infos
        probability = predictions[0][predicted_index]

        return f"{predicted_class_name} (confiance : {probability:.2f}) via Simple CNN"
