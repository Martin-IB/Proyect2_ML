import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
from funciones import extraer_features, reducir_dimension, encontrar_similares,descargar_drive
from PIL import Image




# Cargar los archivos
import joblib
import numpy as np
import pandas as pd

# Descargar archivos desde Google Drive

descargar_drive("1-KMLMO3owErX0UMi5ZffQUWURKPjQipn", "modelo_pca.pkl")
descargar_drive("1YJxDxhH0-ZsmQdOnGtbgcUs0X88U79TN", "X_pca.npy")
descargar_drive("1V31Lf-rmFHB76GSupvW177fiShCw7Dgg", "MovieGenre.csv")
descargar_drive("1UQ51xvxfRymYwrXOoMluBxma6auU8sJp", "movie_features.csv")

pca_model = joblib.load("modelo_pca.pkl")
X_pca = np.load("X_pca.npy")
df = pd.read_csv("MovieGenre.csv", encoding='latin1')

# Cargar el modelo PCA y X_pca
pca_model = joblib.load("modelo_pca.pkl")
X_pca = np.load("X_pca.npy")

# Cargar el dataset
df = pd.read_csv("MovieGenre.csv", encoding='latin1')

# Streamlit UI
st.title("🎬 Movie Recomender System")

uploaded_file = st.file_uploader("Sube el poster de una película", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", width=250)

    # Guardar imagen temporalmente
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Extraer y reducir características
    features = extraer_features("temp.jpg")
    features_reduced = reducir_dimension(features, pca_model)

    # Recomendaciones
    titles = df['Title'].tolist()
    recomendaciones = encontrar_similares(features_reduced, X_pca, df, top_k=5)

    st.subheader("🎥 Recommended Movie Images:")

    recomendaciones_df = pd.DataFrame( recomendaciones)


    cols = st.columns(len(recomendaciones_df))

    for col, (_, row) in zip(cols, recomendaciones_df.iterrows()):
        with col:
            st.write(f"**{row['Title']}**")
            st.markdown(f"*Género:* {row['Genre']}")
            poster_url = row['Poster']
            if isinstance(poster_url, str) and poster_url.startswith("http"):
                st.image(poster_url, width=150)
            else:
                st.warning(f"No hay URL válida para el poster de {row['Title']}")

else:
    st.info("Por favor, sube una imagen para obtener recomendaciones.")


