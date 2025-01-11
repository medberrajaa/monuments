from ultralytics import YOLO

class YOLOv8Model:
    def __init__(self):
        # Charger le modèle YOLOv8 avec vos poids
        self.model = YOLO("weights/best.pt")

    def predict(self, image_path):
        # Effectuer la détection
        results = self.model(image_path)  # Retourne une liste de 'Results'
        
        # On récupère le premier 'Result' s'il existe
        if len(results) > 0:
            result = results[0]
        else:
            return "Aucun résultat renvoyé par YOLOv8."

        # Extraire les boîtes détectées
        boxes = result.boxes  # Détections (bboxes, conf, cls)

        if boxes is None or len(boxes) == 0:
            return "Aucun monument détecté via YOLOv8"

        # Récupérer la classe la plus confiante (exemple)
        # 'cls' sont les indices de classe, 'conf' les confiances
        top_box = max(boxes, key=lambda b: b.conf)  # trouver la box à la plus forte confiance

        class_index = int(top_box.cls.item())  # indice de la classe prédite
        confidence = float(top_box.conf.item())  # confiance associée
        class_name = result.names[class_index]   # nom de la classe, ex: "HassanTower"

        return f"{class_name} (confiance : {confidence:.2f}) via YOLOv8"
