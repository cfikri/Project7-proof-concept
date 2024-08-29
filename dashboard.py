import src.mytools as mt
import streamlit as st
import numpy as np
import pandas as pd
import mlflow
from PIL import Image
import os
import boto3
from sklearn.preprocessing import MinMaxScaler

# Crée une session boto3 pour interagir avec AWS
session = boto3.Session(
    aws_access_key_id=os.getenv("AKIA47CRYSWPLHXWDH4B"),
    aws_secret_access_key=os.getenv("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"),
    region_name=os.getenv("eu-west-1")
)

# Accès au bucket S3
s3 = session.resource('s3')
bucket = s3.Bucket('mlflow-cfikri')

# Configuration
st.set_page_config(page_title="Dashboard ECG", layout="wide")

# Titre de l'application
st.title("Dashboard de Classification d'ECG")

# Affichage des graphiques
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader("Distribution des séries temporelles par classe")
    barplot_image = Image.open("outputs/barplot_beat_classes.png")
    st.image(barplot_image, use_column_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Moyenne des Séries avec Écart-type")
    with open('outputs/mean_std_classes.html', 'r') as file:
        mean_std = file.read()
    st.components.v1.html(mean_std, height=600, scrolling=True)  # Utiliser st.components.v1.html pour afficher le HTML

with col2:
    st.subheader("Représentation 2D des Classes (t-SNE)")
    with open('outputs/visualisation_2D_classes.html', 'r') as file:
        tsne = file.read()
    st.components.v1.html(tsne, height=600, scrolling=True)  # Utiliser st.components.v1.html pour afficher le HTML

# Chargement du modèle depuis MLflow
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://ec2-52-209-5-80.eu-west-1.compute.amazonaws.com:5000"))
    model_uri = "s3://mlflow-cfikri/0/6ef348d774dc484294f4123be9087ea9/artifacts/InceptionTime"
    return mlflow.tensorflow.load_model(model_uri)

model = load_model()

# Dictionnaire de correspondance des classes
class_labels = {
    0: "Normal beat (0)",
    1: "Supraventricular premature beat (1)",
    2: "Premature ventricular contraction (2)",
    3: "Fusion of ventricular and normal beat (3)",
    4: "Unclassifiable beat (4)"
}

# Fonction de prédiction
def predict_class(serie):
    serie = np.array(serie).reshape(-1, 6)
    scaler = MinMaxScaler()
    serie = scaler.fit_transform(serie).flatten()
    subsequences = mt.create_subsequences(serie)
    print("Shape of subsequences for prediction:", subsequences.shape)
    subsequences = np.expand_dims(subsequences, axis=0)
    prediction = model.predict(subsequences)
    class_index = np.argmax(prediction)
    return class_labels[class_index]

# Interface utilisateur pour la saisie de série temporelle
st.header("Quel est le type du signal ?")

# Choisir entre la saisie manuelle ou le téléchargement d'un fichier CSV
option = st.radio(
    "Comment voulez-vous entrer la série temporelle ?",
    ('Saisie manuelle', 'Charger un fichier CSV')
)

if option == 'Saisie manuelle':
    input_serie = st.text_area("Saisir les 192 valeurs de la série, séparées par des virgules")

    if st.button("Prédire"):
        if input_serie:
            try:
                serie = [float(i) for i in input_serie.split(',') if i.strip()]
                if len(serie) != 192:
                    st.error("Le nombre de valeurs saisies doit être exactement de 192.")
                else:
                    st.write(f"La série temporelle est de type : {predict_class(serie)}")
            except ValueError:
                st.error("Veuillez saisir des valeurs numériques séparées par des virgules.")
        else:
            st.error("Veuillez entrer une série temporelle.")

elif option == 'Charger un fichier CSV':
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Lire le fichier CSV
            data = pd.read_csv(uploaded_file, header=None)
            serie = data.values.flatten()  # Convertir en une seule série
            if len(serie) != 192:
                st.error("La série temporelle doit contenir exactement 192 valeurs.")
            else:
                st.write("Série temporelle chargée avec succès:", serie)
                predicted_class = predict_class(serie)
                st.write(f"La série temporelle est de type : {predicted_class}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")