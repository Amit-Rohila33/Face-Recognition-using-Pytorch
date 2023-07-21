# main.py

from src.data_loader import FaceDataset
from src.face_detection import load_face_detection_model, detect_faces
from src.face_feature_extractor import load_face_feature_extractor, extract_features
from src.face_recognition import load_embeddings, recognize_face
from src.utils import load_image, preprocess_image

def main():
    data_dir = 'data/labeled_faces_in_wild'
    test_image_path = 'path/to/test_image.jpg'

    # Load models
    face_detection_model = load_face_detection_model()
    face_feature_extractor = load_face_feature_extractor()

    # Load embeddings
    embeddings = load_embeddings(data_dir, face_feature_extractor)

    # Real-time face recognition on test image
    test_image = load_image(test_image_path)
    faces = detect_faces(face_detection_model, test_image)

    for face in faces:
        preprocessed_face = preprocess_image(face)
        recognized_label, similarity = recognize_face(embeddings, face_feature_extractor, preprocessed_face)

        # Print recognized label and similarity score
        print(f"Recognized Label: {recognized_label}, Similarity Score: {similarity}")

if __name__ == '__main__':
    main()
