import json
import re
import os

# Paths (make sure these match your project)
CAPTIONS_FILE = os.path.join("annotations", "captions_train2014.json")
IMAGES_DIR = "train2014"

def preprocess_caption(caption):
    """Lowercase, remove punctuation, add start/end tokens."""
    caption = caption.lower()
    caption = re.sub(r"[^a-zA-Z0-9\s]", "", caption)
    caption = "<start> " + caption.strip() + " <end>"
    return caption

def load_captions(captions_file=CAPTIONS_FILE):
    """Load captions from JSON file and preprocess them."""
    with open(captions_file, 'r') as f:
        data = json.load(f)

    captions_dict = {}
    for annot in data['annotations']:
        img_id = annot['image_id']
        caption = preprocess_caption(annot['caption'])

        # COCO format: COCO_train2014_000000391895.jpg
        img_filename = f"COCO_train2014_{img_id:012d}.jpg"

        if img_filename not in captions_dict:
            captions_dict[img_filename] = []

        captions_dict[img_filename].append(caption)

    return captions_dict

if __name__ == "__main__":
    captions = load_captions()
    print(f"Loaded {len(captions)} images with captions.")
    # Example check
    first_img = list(captions.keys())[0]
    print(first_img, captions[first_img])
