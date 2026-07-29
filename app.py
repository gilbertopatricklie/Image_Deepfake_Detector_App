import streamlit as st
import keras
import numpy as np
import cv2
from PIL import Image

IMG_SIZE  = (224, 224)
THRESHOLD = 0.515
CROP_MARGIN = 1.25

st.set_page_config(
    page_title="Deteksi Gambar Deepfake",
    layout="wide",
)


@st.cache_resource
def load_model():
    return keras.models.load_model("v18/DFDetect_model.keras", compile=False)


MIN_FACE_SCORE = 5.0  


@st.cache_resource
def load_face_detector():
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_cascade_path  = cv2.data.haarcascades + "haarcascade_eye.xml"
    return cv2.CascadeClassifier(face_cascade_path), cv2.CascadeClassifier(eye_cascade_path)


model = load_model()
face_detector = load_face_detector()


def detect_and_crop_face(pil_img, margin=CROP_MARGIN, min_score=MIN_FACE_SCORE):
    face_cascade, eye_cascade = face_detector

    img_array = np.array(pil_img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces, _, level_weights = face_cascade.detectMultiScale3(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(60, 60),
        outputRejectLevels=True,
    )

    if len(faces) == 0:
        return pil_img, False

    # Saring kandidat dengan confidence (level_weight) di bawah ambang,
    # supaya false positive kecil tidak ikut dipertimbangkan
    candidates = [(f, s) for f, s in zip(faces, level_weights) if s >= min_score]
    if not candidates:
        return pil_img, False

    # Cross-check tiap kandidat: harus ada minimal 1 mata terdeteksi di dalamnya
    validated = []
    for (x, y, w, h), score in candidates:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
        )
        if len(eyes) >= 1:
            validated.append(((x, y, w, h), score))

    # Fallback ke candidates kalau eye-check terlalu ketat (mis. kacamata gelap)
    pool = validated if validated else candidates
    (x, y, w, h), _ = max(pool, key=lambda c: c[1])  # pilih confidence tertinggi, bukan area terbesar

    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * margin, h * margin

    img_w, img_h = pil_img.size
    left   = max(0, int(cx - new_w / 2))
    top    = max(0, int(cy - new_h / 2))
    right  = min(img_w, int(cx + new_w / 2))
    bottom = min(img_h, int(cy + new_h / 2))

    cropped = pil_img.crop((left, top, right, bottom))
    return cropped, True


def preprocess_for_model(pil_img):
    img_resized = pil_img.resize(IMG_SIZE, resample=Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("Tentang Aplikasi")

    with st.expander("Model yang digunakan", expanded=True):
        st.markdown(
            """
            **Backbone:** EfficientNetV2B0 (transfer learning, fine-tuned)

            **Dataset latih:** FaceForensics++ (c23) — kelas Real
            serta 5 jenis manipulasi (Deepfakes, Face2Face, FaceShifter,
            FaceSwap, NeuralTextures)

            Wajah pada gambar di-*crop* otomatis sebelum
            masuk ke model, mengikuti metodologi preprocessing FaceForensics++.
            """
        )

    with st.expander("Apa itu gambar deepfake?"):
        st.markdown(
            """
            Gambar deepfake adalah gambar buatan yang dihasilkan atau
            dimanipulasi dengan Artificial Intelligence untuk
            mengganti wajah, ekspresi atau hal lainnya.

            Alat deteksi ini membantu memberi indikasi, tapi
            hasilnya bukan merupakan bukti forensik mutlak — terutama pada gambar yang
            dikompresi berat atau teknik pembuatan deepfake yang baru.
            """
        )

    st.divider()
    st.caption(f"Threshold klasifikasi yang dipakai: **{THRESHOLD}**")

# ───────────────────────── Main ─────────────────────────
st.title("Deepfake Face Detector")

uploaded = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    original_img = Image.open(uploaded).convert("RGB")

    with st.spinner("Mendeteksi wajah dan menjalankan model..."):
        cropped_img, face_found = detect_and_crop_face(original_img)

        if not face_found:
            st.warning(
                "Wajah tidak terdeteksi secara otomatis. Hasil prediksi mungkin kurang "
                "akurat karena model dilatih khusus pada gambar wajah yang sudah di-crop. "
                "Coba upload foto dengan wajah yang lebih jelas/frontal."
            )

        arr = preprocess_for_model(cropped_img)
        prob_fake = float(model.predict(arr, verbose=0)[0][0])
        prob_real = 1 - prob_fake

    st.divider()

    # Gambar asli & hasil crop side-by-side, supaya user bisa verifikasi
    # apakah crop wajah berhasil dengan benar
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(original_img, caption="Gambar Asli", use_container_width=True)
    with col_img2:
        st.image(cropped_img, caption="Wajah Terdeteksi (input ke model)", use_container_width=True)

    st.divider()

    if prob_fake >= THRESHOLD:
        st.error(f"FAKE  ({prob_fake*100:.2f}%)")
    else:
        st.success(f"REAL  ({prob_real*100:.2f}%)")

    col1, col2 = st.columns(2)
    col1.metric("Fake Probability", f"{prob_fake*100:.2f}%")
    col2.metric("Real Probability", f"{prob_real*100:.2f}%")

    st.markdown("**Confidence:**")
    st.progress(prob_fake, text=f"Fake confidence: {prob_fake*100:.1f}%")

    st.caption(
        f"Threshold aktif: {THRESHOLD} — "
        f"gambar dengan probabilitas Fake ≥ {THRESHOLD*100:.0f}% "
        f"akan diklasifikasikan sebagai FAKE."
    )

    st.info(
        "Hasil ini merupakan output model machine learning dan bersifat "
        "indikatif, bukan bukti forensik yang pasti.",
    )
