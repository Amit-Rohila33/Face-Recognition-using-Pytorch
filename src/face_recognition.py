# face_recognition.py

import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(data_dir, face_feature_extractor):
    embeddings = {}
    dataset = FaceDataset(data_dir)
    for i, (image, label) in enumerate(dataset):
        face = torch.unsqueeze(image, 0)
        with torch.no_grad():
            features = extract_features(face_feature_extractor, face)
        embeddings[i] = features
    return embeddings

def recognize_face(embeddings, face_feature_extractor, face):
    face_features = extract_features(face_feature_extractor, face)
    similarities = {}
    for i, embedding in embeddings.items():
        similarity = cosine_similarity(embedding.reshape(1, -1), face_features.reshape(1, -1))
        similarities[i] = similarity[0][0]
    recognized_label = max(similarities, key=similarities.get)
    return recognized_label, similarities[recognized_label]
