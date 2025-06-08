# Recomendador de peliculas

Este proyecto es una aplicación web para recomendar películas a partir de una imagen de póster, utilizando reducción de dimensionalidad (PCA) y características extraídas de imágenes. La app está desarrollada con Streamlit.


Desde el terminal de consola seguir los siguientes pasos:

1: clonar la carpeta del proyecto desde hithub 

git clone https://github.com/Martin-IB/nuevo_app.git

2: Acceder a la carpeta que contenga el poryecto

cd  Proyect2_ML

3: crear un entorno virtual 
py -m venv venv

4: activar el entono virual
venv\Scripts\activate

5: Instalar los requerimientos 
pip install -r requirements.txt

6: Ejecutar la aplicacion
streamlit run app.py

7: Copiar y pegar el URL que aparece en el terminal en algun navegador web, ejemplo:
http://localhost:8000

Esperar a que se descarguen los modelos entrenados para ver los resultados
8: Adicionales
-Verificar la version scikit-learn debido a que el modelo fue entrenado con la version scikit-learn==1.7.0
-Sino instalar la version 
pip install scikit-learn==1.7.0
