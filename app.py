import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Import des modèles
from models.simple_cnn import SimpleCNNModel
from models.resnet import ResNetModel
from models.yolov8 import YOLOv8Model

app = Flask(__name__)

# Dossier où stocker les images uploadées
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Instanciation des 3 modèles (chacun charge ses poids dans son __init__)
simple_cnn = SimpleCNNModel()
resnet = ResNetModel()
yolov8 = YOLOv8Model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer le modèle sélectionné
        selected_model = request.form.get('model_select')
        # Récupérer le fichier image
        file = request.files.get('image')

        if file and file.filename != '':
            # Sécuriser le nom de fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Choisir le modèle en fonction de l'option sélectionnée
            if selected_model == 'simple_cnn':
                prediction = simple_cnn.predict(filepath)
            elif selected_model == 'resnet':
                prediction = resnet.predict(filepath)
            else:
                prediction = yolov8.predict(filepath)

            # (Optionnel) Supprimer l'image après prédiction
            # os.remove(filepath)

            return render_template('index.html',
                                   prediction=prediction,
                                   filename=filename,
                                   model_selected=selected_model)

        # Si aucune image n'est chargée, on revient à l'accueil
        return redirect(url_for('index'))

    # Méthode GET : on affiche simplement la page
    return render_template('index.html')

if __name__ == '__main__':
    # Pour Docker, on peut mettre host='0.0.0.0' si on veut l'exposer à l'exterieur.
    app.run(debug=True, host='0.0.0.0', port=5000)
