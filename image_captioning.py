import os
import pickle
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# Load preprocessed data
with open('features/image_features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('annotations/captions_train2014.json', 'r') as f:
    captions_data = json.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# Step 1: Build training caption list for max_length
train_captions = []
for annot in captions_data['annotations']:
    caption = f"startseq {annot['caption']} endseq"
    train_captions.append(caption)

# Step 2: Correct max_length calculation using original text captions
max_length = max(len(caption.split()) for caption in train_captions)

# Create image-caption mapping (limit to 1000 images)
image_captions = {}
for annot in captions_data['annotations']:
    img_id = f"COCO_train2014_{annot['image_id']:012d}"
    caption = f"startseq {annot['caption']} endseq"
    if img_id in features:
        image_captions.setdefault(img_id, []).append(caption)
    if len(image_captions) >= 1000:
        break

# Create training data
def create_sequences(tokenizer, max_length, captions_list, feature, vocab_size):
    X1, X2, y = [], [], []
    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = np.eye(vocab_size)[out_seq]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Prepare training dataset
X1_train, X2_train, y_train = [], [], []
for img_id, caps in image_captions.items():
    feature = features[img_id]
    input1, input2, output = create_sequences(tokenizer, max_length, caps, feature, vocab_size)
    X1_train.extend(input1)
    X2_train.extend(input2)
    y_train.extend(output)

X1_train = np.array(X1_train)
X2_train = np.array(X2_train)
y_train = np.array(y_train)

print(f"🧠 Training on {len(X1_train)} samples...")

# Define the model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

# Save best model
checkpoint = ModelCheckpoint('model.h5', monitor='loss', save_best_only=True, verbose=1)

# Train the model
model.fit([X1_train, X2_train], y_train, epochs=20, batch_size=64, callbacks=[checkpoint])

print("✅ Model training completed and saved as model.h5")
