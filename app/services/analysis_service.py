from __future__ import annotations

from typing import List
import mediapipe as mp
import numpy as np
from app.ml.faces.recognizer import FaceRecognizer
from app.ml.classifiers.goal_clip_keras import GoalNoGoalClassifier
from app.ml.detectors.jerseys import JerseyDetector
from app.schemas.io import FacePrediction, EventPrediction, JerseyDetection, CompleteResponse

# A pipeline for face detection, by the mediapipe framework from google.
mp_face_detection = mp.solutions.face_detection

class AnalysisService:
    
    # Nuestro constructor recibe objetos que son modelos de ml,
    # concretamente los que estan dentro de la carpeta ml
    def __init__(self, face_rec: FaceRecognizer,
                 goal_clf: GoalNoGoalClassifier,
                 jersey_det: JerseyDetector):
        self.face_rec = face_rec
        self.goal_clf = goal_clf
        self.jersey_det = jersey_det

    # Analyze nos restorna un objeto de tipo CompleteResponse
    # que es sencillamente un modelo de pydantic
    def analyze(self, img_pil) -> CompleteResponse:
        from app.utils.images import pil_to_rgb_numpy
        
        # Primero transformamos nuestra imagen de formato pil a rgb
        img_rgb = pil_to_rgb_numpy(img_pil)

        # 1) Detección de caras (MediaPipe)
        faces_out: List[FacePrediction] = []
        bboxes_trbl, det_scores = [], []
        
        # Media pipe nos detecta las bounding boxes, trabaja con imagenes rgb y nos retorna nuestros resultados,
        # en coordinadas normalizadas.
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
            res = fd.process(img_rgb)
        
        # Si hay una respuesta, examinamos nuestra respuesta (hablando de la respuesta del pipeline)    
        if res and res.detections:
            
            h, w = img_rgb.shape[:2] # altura y ancho de la imagen
            
            for det in res.detections:
                
                # Convertimos las coordenadas a coordenadas pixel
                rb = det.location_data.relative_bounding_box
                
                x, y = int(rb.xmin*w), int(rb.ymin*h)
                ww, hh = int(rb.width*w), int(rb.height*h)
                x, y = max(0,x), max(0,y)
                x2, y2 = min(w-1, x+ww), min(h-1, y+hh)
                top, right, bottom, left = y, x2, y2, x
                
                # detection -> son la lista de confianza de los puntajes de detección
                # bboxes_trbl -> lista de las bounding boxes de las caras detectadas
                
                bboxes_trbl.append([top,right,bottom,left])
                det_scores.append(float(det.score[0]) if det.score else 0.0)

        # 1.2) Embeddings + clasificación
        labels_scores = []
        
        # Si tenemos bounding boxes, es decir hay caras detectadas.
        if bboxes_trbl:
            # obtenemos nuestra matriz de embeddings que nos da nuestro código de FaceRecognizer
            embs = self.face_rec.encodings(img_rgb, bboxes_trbl)
            # luego utilizamos el método de clasificar y guardamos cada clase con su puntaje de confianza
            labels_scores = self.face_rec.classify(embs)

        # recorremos nuestros bounding boxes con un indice y un enum
        for i, trbl in enumerate(bboxes_trbl):
            # si el indice está dentro de lo esperado podemos 
            if i < len(labels_scores):
                # (nombre de la clase, puntaje de confianza)
                lbl, s = labels_scores[i]
                faces_out.append(FacePrediction(bbox=list(map(int, trbl)), label=str(lbl), score=float(s)))
            else:
                faces_out.append(FacePrediction(bbox=list(map(int, trbl)), label="unknown",
                                                score=float(det_scores[i]) if i < len(det_scores) else 0.0))

        # 2) Goal/NoGoal
        # predecimos que tipo de evento puede tener la imagen que recibimos mediante el goal no goal clf
        preds = self.goal_clf.predict(img_rgb, top_k=2)
        
        # si tenemos predicciones avanzamos
        if preds:
            # pasamos la información de los diccionarios (eventos predichos) a los modelos de pydantic 
            event_predictions = [EventPrediction(**p) for p in preds]
            
            # guardamos el evento más probable
            top_event = event_predictions[0]
        else:
            event_predictions = []
            top_event = EventPrediction(event_class="Unknown", confidence=0.0, percentage=0.0)

        # 3) Camisetas - CORREGIDO: ahora sí define las variables
        # Obtenemos un arreglo de camisetas JerseyDetection que son modelos de pydantic
        jerseys_raw = self.jersey_det.detect(img_rgb)
        
        # Nos aseguramos de que los objetos jersey que detectamos no es un diccionario vacío
        jerseys = [JerseyDetection(**j) if isinstance(j, dict) else j for j in jerseys_raw]
        
        # Contamos la cantidad de jerseys de un equipo u otro detectado
        a_cnt = sum(1 for j in jerseys if j.team == "Argentina")
        f_cnt = sum(1 for j in jerseys if j.team == "France")

        # Construimos nuestra respuesta
        return CompleteResponse(
            num_faces=len(faces_out),
            faces=faces_out,
            event_predictions=event_predictions,
            top_event=top_event,
            jerseys=jerseys,
            argentina_count=a_cnt,
            france_count=f_cnt,
            image_processed=True,
            total_detections=len(faces_out) + len(jerseys),
        )
    
    def mock_analysis(detected_event):        
        return detected_event.event_type