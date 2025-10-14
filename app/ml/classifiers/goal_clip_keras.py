import numpy as np, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tensorflow.keras.models import load_model

class GoalNoGoalClassifier:
    # Recibimos el path del modelo, y el path de las etiquetas que tenemos que predecir "creo"
    # de momento no se que es el device.
    def __init__(self, model_path: str, labels_path: str, device: str = None):
        
        # Escogemos el dispositivo de computo (GPU) para realizar los calculos de la clasificación, si lo hay, si no lo hay se deja por
        # defecto la cpu.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargamos una red neuronal pre entrenada que permite convertir imagenes a embeddings, representando el contenido semántico.
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        # Cargamos el procesador de openai, que nos permite hacer calculos adicionales, como normalización y transformación. Esta es una clase.
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Cargamos el modelo usando la función load model, mediante el path donde está el modelo.
        self.model = load_model(model_path)
        
        # Seteamos las clases usando la función load de numpy que son las clases que nosotros le pasamos.
        self.classes_ = np.load(labels_path, allow_pickle=True)

    # Esta es una función auxiliar que recibe una imagen en formato rgb, y la convierte a formato PIL.
    def _clip_embedding(self, img_rgb: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img_rgb) # Conversión a PIL, Image.fromarray() just wraps it in order to make it understandable by CLIPProcessor.
        
        # Aqui usamos nuestro CLIPProcessor.
        # Cambia su tamaño al esperado de entrada 224px * 224px
        # Normaliza los valores de los pixeles.
        # Convierte nuestro arreglo de números en un tensor de pytorch.
        # finalmente mueve nuestro tensor a la GPU o la CPU si está disponible.
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        
        # Esta operación garantiza que no se calculen los gradientes, ya que queremos ahorrar memoria y tampoco estamos entrenando, solo haciendo inferencia.
        with torch.no_grad():
            # La función get_image_features pasa las imagenes pre procesadas a través del codificador de visión de CLIP.
            emb = self.clip.get_image_features(**inputs)
            # Obtenemos emb que es un embedding de la imagen, y es un formato de pytorch. 
            
        # Finalmente convertimos el tensor a un arreglo de numpy.
        return emb[0].detach().cpu().numpy().astype(np.float32)

    
    def predict(self, img_rgb: np.ndarray, top_k=2):
        # Llamamos a clip embeding que es la función de arriba y luego aplicamos un reshape produciendo un arreglo de 
        # (1, n)
        emb = self._clip_embedding(img_rgb).reshape(1, -1)
        
        # Llamamos al modelo y le damos a predecir (el modelo que cargamos previamente.)
        # Predict retorna una distribución de probabilidad sobre las posibles clases a las que puede pertenecer nuestra imagen:
        # goal vs no goal.
        proba = self.model.predict(emb, verbose=0)[0]
        
        # Con el indice [0] accedemos a las predicciones de las clases (son probabilidades).
        
        # Obtenemos los indices de las probabilidades en orden descendente y cogemos las primeras 2: "top-k"
        idx = np.argsort(proba)[::-1][:top_k]
        
        # Retornamos un arreglo de diccionarios, que contienen la clase del evento clasificato, la confianza en la predcción y el porcentace
        # de esa confianza.
        return [{"event_class": str(self.classes_[i]),
                 "confidence": float(proba[i]),
                 "percentage": float(proba[i]*100)} for i in idx]
