from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PATH CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "crowddensity_yolo11n_full_best.pt"
RESULT_DIR = ROOT / "assets" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CrowdDensity Mini",
    page_icon="👥",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan:\n{MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


try:
    model = load_model()

except Exception as e:
    st.error("Model gagal dimuat.")
    st.code(str(e))
    st.stop()


# ============================================================
# UI HEADER
# ============================================================

st.title("👥 CrowdDensity Mini")

st.caption(
    "YOLO11n person detection + crowd counting baseline "
    "trained on the full annotated CrowdHuman dataset."
)

st.info(
    "Model ini adalah person detection + counting baseline, "
    "bukan density-map estimator."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Inference Settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.divider()

    st.subheader("Model")

    st.write("**Architecture:** YOLO11n")
    st.write("**Task:** Person Detection")
    st.write("**Image Size:** 640")
    st.write("**Training Images:** 15,000")
    st.write("**Validation Images:** 4,370")

    st.divider()

    st.subheader("Validation")

    st.metric("Precision", "83.70%")
    st.metric("Recall", "69.68%")
    st.metric("mAP50", "80.92%")
    st.metric("mAP50-95", "48.90%")

    st.caption(
        "Metrics berasal dari CrowdHuman validation set."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload gambar kerumunan",
    type=["jpg", "jpeg", "png", "webp"],
)


if uploaded_file is None:

    st.markdown(
        """
        ### Cara menggunakan

        1. Upload gambar.
        2. Atur confidence threshold jika diperlukan.
        3. Klik **Analyze Crowd**.
        4. Model akan mendeteksi dan menghitung orang.
        """
    )

    st.stop()


# ============================================================
# LOAD INPUT IMAGE
# ============================================================

try:
    image = Image.open(uploaded_file).convert("RGB")

except Exception as e:

    st.error("File gambar tidak dapat dibaca.")
    st.code(str(e))
    st.stop()


# ============================================================
# IMAGE PREVIEW
# ============================================================

left, right = st.columns(2)

with left:

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True,
    )


# ============================================================
# RUN INFERENCE
# ============================================================

if st.button(
    "🔍 Analyze Crowd",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Detecting people..."):

        try:

            image_np = np.array(image)

            result = model.predict(
                source=image_np,
                imgsz=640,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=400,
                verbose=False,
            )[0]

        except Exception as e:

            st.error("Inference gagal.")
            st.code(str(e))
            st.stop()


    # ========================================================
    # EXTRACT DETECTIONS
    # ========================================================

    boxes = result.boxes

    detections = []

    if boxes is not None:

        for box in boxes:

            xyxy = box.xyxy[0].cpu().tolist()

            confidence = float(
                box.conf[0].cpu().item()
            )

            class_id = int(
                box.cls[0].cpu().item()
            )

            class_name = result.names.get(
                class_id,
                str(class_id),
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [
                        round(float(v), 2)
                        for v in xyxy
                    ],
                }
            )


    # ========================================================
    # COUNT PEOPLE
    # ========================================================

    person_detections = [
        det
        for det in detections
        if det["class_name"].lower() == "person"
    ]

    person_count = len(person_detections)


    # ========================================================
    # CONFIDENCE SUMMARY
    # ========================================================

    if person_detections:

        confidences = [
            item["confidence"]
            for item in person_detections
        ]

        avg_confidence = float(
            np.mean(confidences)
        )

        max_confidence = float(
            np.max(confidences)
        )

    else:

        avg_confidence = 0.0
        max_confidence = 0.0


    # ========================================================
    # RENDER YOLO RESULT
    # ========================================================

    plotted_bgr = result.plot()

    plotted_rgb = cv2.cvtColor(
        plotted_bgr,
        cv2.COLOR_BGR2RGB,
    )

    result_image = Image.fromarray(
        plotted_rgb
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    image_output_path = (
        RESULT_DIR
        / f"crowd_result_{timestamp}.jpg"
    )

    json_output_path = (
        RESULT_DIR
        / f"crowd_result_{timestamp}.json"
    )

    result_image.save(
        image_output_path,
        quality=95,
    )


    result_payload = {
        "timestamp": timestamp,
        "source_filename": uploaded_file.name,
        "model": MODEL_PATH.name,
        "task": "person_detection_and_counting",
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "detected_person_count": person_count,
        "average_detection_confidence": round(
            avg_confidence,
            6,
        ),
        "maximum_detection_confidence": round(
            max_confidence,
            6,
        ),
        "detections": person_detections,
        "limitations": (
            "Person detection and counting baseline. "
            "This is not a crowd density-map estimator. "
            "Severe occlusion, tiny persons, threshold settings, "
            "and domain shift may affect the count."
        ),
    }


    with open(
        json_output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result_payload,
            f,
            indent=2,
        )


    # ========================================================
    # RESULT IMAGE
    # ========================================================

    with right:

        st.subheader("Detection Result")

        st.image(
            result_image,
            use_container_width=True,
        )


    # ========================================================
    # METRICS
    # ========================================================

    st.divider()

    st.subheader("📊 Detection Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Detected People",
        person_count,
    )

    col2.metric(
        "Average Confidence",
        f"{avg_confidence * 100:.2f}%"
        if person_count > 0
        else "—",
    )

    col3.metric(
        "Highest Confidence",
        f"{max_confidence * 100:.2f}%"
        if person_count > 0
        else "—",
    )


    # ========================================================
    # DETECTION TABLE
    # ========================================================

    if person_detections:

        st.subheader("Detected Persons")

        detection_rows = []

        for index, det in enumerate(
            person_detections,
            start=1,
        ):

            detection_rows.append(
                {
                    "Person": index,
                    "Confidence": (
                        f"{det['confidence'] * 100:.2f}%"
                    ),
                    "Bounding Box": (
                        str(det["bbox_xyxy"])
                    ),
                }
            )

        st.dataframe(
            detection_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "Tidak ada person yang terdeteksi "
            "pada confidence threshold saat ini."
        )


    # ========================================================
    # SAVED OUTPUT
    # ========================================================

    st.success(
        "Inference selesai dan hasil berhasil disimpan."
    )

    st.caption(
        f"Image: {image_output_path.relative_to(ROOT)}"
    )

    st.caption(
        f"JSON: {json_output_path.relative_to(ROOT)}"
    )


    # ========================================================
    # LIMITATION
    # ========================================================

    st.divider()

    st.warning(
        """
        **Interpretasi hasil**

        Jumlah yang ditampilkan adalah jumlah bounding box `person`
        yang berhasil dideteksi model pada threshold yang dipilih.

        Angka ini bukan ground-truth jumlah orang dan dapat mengalami
        under-count atau over-count, terutama pada kondisi crowd padat,
        occlusion berat, orang berukuran sangat kecil, atau domain
        gambar yang berbeda dari CrowdHuman.

        CrowdDensity Mini saat ini merupakan **person detection +
        counting baseline**, bukan density-map crowd estimator.
        """
    )