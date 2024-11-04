import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

# Memuat model
try:
    model = tf.keras.models.load_model('MODEL/ssd_sibi_model_mediapipe.h5')
except Exception as e:
    st.error("Model tidak dapat dimuat. Pastikan path model benar dan model tersedia.")
    st.stop()

# Fungsi untuk memproses gambar
def load_and_preprocess_image(image, target_size=(300, 300)):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array /= 255.0  # Normalisasi
    return img_array

# Fungsi untuk deteksi gambar statis
def detect_static_image(uploaded_file):
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    
    img_array = load_and_preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Definisikan class_labels sesuai dengan kelas Anda
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    if len(class_labels) != predictions.shape[1]:
        st.error("Jumlah kelas pada class_labels tidak sesuai dengan output model.")
    else:
        predicted_label = class_labels[predicted_class[0]]
        st.write(f"Predicted label: {predicted_label}")

def detect_realtime():
    st.write("Menggunakan Kamera untuk Deteksi Real-time")

    # Membuka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera tidak dapat diakses.")
        return

    # Menampilkan kamera secara real-time
    stframe = st.empty()
    stop_button = st.checkbox("Stop Real-time Detection")  # Tombol 'Stop' dalam bentuk checkbox

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Tidak dapat mengakses frame.")
            break

        # Mengubah ukuran gambar sesuai dengan input model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (300, 300))
        img_array = np.expand_dims(resized_frame / 255.0, axis=0)
        
        # Melakukan prediksi
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        predicted_label = class_labels[predicted_class[0]]

        # Menampilkan hasil prediksi pada frame
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Judul aplikasi
st.title("Prediksi Menggunakan Model SSD")

# Pilihan deteksi
option = st.radio("Pilih Mode Deteksi", ('Deteksi Gambar', 'Deteksi Real-time'))

# Menjalankan pilihan deteksi
if option == 'Deteksi Gambar':
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png"])
    if uploaded_file is not None:
        detect_static_image(uploaded_file)

elif option == 'Deteksi Real-time':
    detect_realtime()
