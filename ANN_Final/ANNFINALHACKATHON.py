# Since coding the entire solution for the given task is extensive, I'll provide a structured outline along with code snippets
# that can be used as a starting point for each phase. The user would need to expand upon these snippets based on their specific environment and data.

# Required Libraries
import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Phase 1: Model Training

# Assumption: The training data is already segregated into folders for each category
# and the images are named appropriately for easy loading.

# Directory setup (example paths, need to be adapted)
train_data_dir = r'C:\Program Files\deneme\train_data'
categories = ['big_face_celebrityA', 'small_face_celebrityA', 'big_face_celebrityB', 'small_face_celebrityB',
              'not_face']

# Hyperparameters
img_size = (64, 64)  # or any other size deemed appropriate
epochs = 10
batch_size = 32
learning_rate = 0.001


# Model Architecture (should be based on the ANN course content)
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Load and preprocess the data
def load_data():
    images = []
    labels = []

    for category in categories:
        path = os.path.join(train_data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_array = cv2.resize(img_array, img_size)
                images.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                pass

    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Normalize image data
    X = X / 255.0

    return X, y


# Split the data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train = np.eye(len(categories))[y_train]
y_test = np.eye(len(categories))[y_test]

# Initialize and train the model
model = create_model()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Plotting training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
model.save('my_model.h5')

# Display confusion matrix
print(conf_matrix)

import cv2
from tensorflow.keras.models import load_model

# Modelinizi yükleyin
model = load_model('my_model.h5')

# OpenCV'nin yüz dedektörünü yükleyin
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(frame, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return faces


def preprocess_face(face, target_size=img_size):
    face = cv2.resize(face, target_size)
    face = face.astype('float32')
    face /= 255
    face = np.expand_dims(face, axis=0)  # Modelin beklediği şekle getir
    return face


# Video dosyasını açın
cap = cv2.VideoCapture(r'C:\Program Files\deneme\Training_Video_Jimmy Kimmel-Jake Johnson.mp4')  # Doğru video yolu

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Yüzleri algıla
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        # Algılanan yüzü ön işlemden geçir
        face = frame[y:y + h, x:x + w]
        preprocessed_face = preprocess_face(face)

        # Model ile tahmin yap
        prediction = model.predict(preprocessed_face)
        class_index = np.argmax(prediction)

        # Eğer tahmin edilen yüz bir ünlüye aitse, etrafına bir kutu çiz
        if prediction[0][class_index] > 0.5:  # Eşik değeri
            label = categories[class_index]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Kareyi göster
    cv2.imshow('Video', frame)

    # 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Videoyu serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np

# OpenCV'nin yüz dedektörünü yükleyin
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yüzleri saklamak için klasör oluşturun
other_faces_dir = 'path_to_others_folder'  # Bu klasörün var olduğundan emin olun
if not os.path.exists(other_faces_dir):
    os.makedirs(other_faces_dir)

def save_face(face, count):
    file_name = f"face_{count}.png"
    file_path = os.path.join(other_faces_dir, file_name)
    cv2.imwrite(file_path, face)

# Videoyu açın
cap = cv2.VideoCapture('path_to_video.mp4')

# Benzersiz yüzleri kaydetmek için bir sayac ve yüz listesi
unique_faces = []
face_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Yüzleri algıla
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        # Yüzü kes
        face = frame[y:y + h, x:x + w]

        # Yüzü benzersiz yüzler listesine ekleyin
        # Not: Burada benzersizliği kontrol etmek için daha karmaşık bir yöntem kullanmanız gerekebilir
        if not any(np.array_equal(face, f) for f in unique_faces):
            unique_faces.append(face)
            save_face(face, face_id)
            face_id += 1

# Videonun serbest bırakılması ve tüm pencerelerin kapatılması
cap.release()
cv2.destroyAllWindows()

# Benzersiz yüz sayısını yazdır
print(f"Number of unique faces detected: {len(unique_faces)}")
