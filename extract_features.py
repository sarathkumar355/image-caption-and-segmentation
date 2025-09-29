import os
import pickle
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image
import numpy as np

# ✅ Paths (match with data_loader.py)
IMAGES_PATH = "train2014"
FEATURES_DIR = "features"
FEATURES_FILE = os.path.join(FEATURES_DIR, "image_features.pkl")

# ✅ Ensure output folder exists
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load pre-trained InceptionV3 model + remove final layer
base_model = InceptionV3(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(limit=None):
    features = {}
    image_files = [f for f in os.listdir(IMAGES_PATH) if f.lower().endswith(".jpg")]
    
    if limit:
        image_files = image_files[:limit]

    for img_name in tqdm(image_files, desc="Extracting features", total=len(image_files)):
        img_path = os.path.join(IMAGES_PATH, img_name)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        feature = model.predict(x, verbose=0)
        feature = np.reshape(feature, feature.shape[1])

        # ✅ Use full filename as key for easy mapping
        features[img_name] = feature

    return features

if __name__ == "__main__":
    print("🚀 Starting feature extraction...")
    features = extract_features(limit=1000)  # Change or remove limit for full run
    
    with open(FEATURES_FILE, "wb") as f:
        pickle.dump(features, f)

    print(f"✅ Extracted and saved features for {len(features)} images to {FEATURES_FILE}")
