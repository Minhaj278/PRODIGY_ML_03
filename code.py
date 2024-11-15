import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data_dir = r"C:\Users\baasi\task3\dogs-vs-cats\train"
img_size = (64, 64)

images = []
labels = []

print("Files in the directory:", os.listdir(data_dir))

for img_name in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_name)
    
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {img_name}...")
        
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_name}")
            continue
        
        img = cv2.resize(img, img_size)
        img = img.flatten()
        images.append(img)
        
        if 'cat' in img_name.lower():
            labels.append('cat')
        elif 'dog' in img_name.lower():
            labels.append('dog')

X = np.array(images)
y = np.array(labels)

print(f"Number of images processed: {len(X)}")
print(f"Number of labels: {len(y)}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
