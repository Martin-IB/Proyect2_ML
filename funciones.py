import numpy as np
from skimage import io, color, transform, feature
from sklearn.metrics.pairwise import euclidean_distances
import os
import requests
import gdown
import joblib
import numpy as np
import pandas as pd


def extract_hsv_histogram(image_path, bins=(8, 12, 3)):
    try:
        image = io.imread(image_path)
        if image.shape[-1] == 4:
            image = image[..., :3]
        hsv = color.rgb2hsv(image)
        hist = np.histogramdd(
            [hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()],
            bins=bins,
            range=((0, 1), (0, 1), (0, 1)))
        return hist[0].ravel() / np.sum(hist[0])
    except:
        return np.zeros(np.prod(bins))

def extract_hog_features(image_path, pixels_per_cell=(16, 16), cells_per_block=(3, 3)):
    try:
        image = io.imread(image_path)
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        resized = transform.resize(image, (128, 64))  
        hog_features = feature.hog(
            resized,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block)
        return hog_features
    except:
        return np.zeros(9 * 4 * 7 * 7)  


def descargar_drive(id_archivo, nombre_local):
    if not os.path.exists(nombre_local):
        url = f"https://drive.google.com/uc?id={id_archivo}"
        print(f"Descargando {nombre_local} desde Google Drive...")
        gdown.download(url, nombre_local, quiet=False, fuzzy=True)
    else:
        print(f"{nombre_local} ya existe. No se descargará de nuevo.")


def extraer_features(image_path):
    hog_feat = extract_hog_features(image_path)

    hsv_feat = extract_hsv_histogram(image_path)

    features = np.concatenate([hog_feat, hsv_feat])
    return features


def reducir_dimension(features, pca_model):
    if features.ndim == 1:
        features = features.reshape(1, -1)  

    return pca_model.transform(features)


def encontrar_similares(features_reduced, X_reduced, df, top_k=5):
    from sklearn.metrics import euclidean_distances

    distancias = euclidean_distances(features_reduced, X_reduced)
    indices_cercanos = distancias.argsort()[0, 1:top_k+1]  # los k más cercanos

    peliculas_recomendadas =df.iloc[indices_cercanos]
    return peliculas_recomendadas


def recomendar_por_imagen(imagen_ruta, X_reduced, titles, pca_model, top_k=5):
    features = extraer_features(imagen_ruta)
    features_reduced = reducir_dimension(features)
    recomendadas = encontrar_similares(features_reduced, X_reduced, titles, top_k)
    return recomendadas
