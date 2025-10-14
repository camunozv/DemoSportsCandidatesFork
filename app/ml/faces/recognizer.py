import json, os, numpy as np, joblib, face_recognition
from typing import List, Tuple

class FaceRecognizer:
    def __init__(self, model_path, scaler_path, labels_json, pca_path=None):
        # This variable is flag that indicates that the FaceRecognizer has been successfully 
        # instantiated.
        self.loaded = False
        
        # This if verifies if the paths that we pass to the FaceRecognizer instance are within our project.
        if all(os.path.exists(p) for p in [model_path, scaler_path, labels_json]):
            self.clf = joblib.load(model_path) # we load the model
            self.scaler = joblib.load(scaler_path) # we load the scaler -> this will be useful later with the embeddings.
            self.pca = joblib.load(pca_path) if (pca_path and os.path.exists(pca_path)) else None # optionally we can load a pca but if no it is ok
            
            # We load our json object, with our classification tags into the meta object.
            with open(labels_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
            # Then we load those tags into the labels object of our FaceRecognizer in the form of an np.array
            # classes = ["Messi", "Ronaldo", "Neymar"]
            self.labels = np.array(meta.get("classes_", meta.get("classes", [])))
            self.loaded = True # Marks everything loaded successfully

    # This functions receives the rgb codification of our image.
    # And also receives a list of bounding boxes in the trbl format (top, right, bottom, left)
    # Face bounding boxes.
    def encodings(self, img_rgb, bboxes_trbl: List[List[int]]) -> np.ndarray:
        
        # We extract the embedings we detected, which are of course the one we pased
        # and we return a matrix with the images stacked
        encs = face_recognition.face_encodings(
            img_rgb, known_face_locations=bboxes_trbl, num_jitters=1, model="small"
        )
        return np.vstack(encs).astype(np.float32) if encs else np.empty((0, 128), np.float32)

    # This funcition is in charge of making the classification giving a conficence score.
    def classify(self, embs: np.ndarray) -> List[Tuple[str, float]]:
        if not self.loaded or embs.size == 0:
            return []
        # Scale the embedings to a god dimension
        X = self.scaler.transform(embs)
        
        # If we have pca we can do a transform in order to improve speed or something
        if self.pca is not None:
            X = self.pca.transform(X)
        
        # We get the probability of belonging to a class, the X consists of two columns
        # one with probabilites, and another with the classes we need.
        proba = self.clf.predict_proba(X)
        
        # Extract the index of the class with the highest probability.
        ids = np.argmax(proba, axis=1)
        return [(str(self.labels[i]) if len(self.labels) else str(i), float(proba[j, i]))
                for j, i in enumerate(ids)]
