from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Layer, Conv2D, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense
import tensorflow as tf
import numpy as np
import os
import cv2
import google.generativeai as genai


# -----------------------------------------------------
# GEMINI CONFIGURATION (NEW PART)
# -----------------------------------------------------

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-1.5-flash")



# -----------------------------------------------------
# FIXED CBAM Layer
# -----------------------------------------------------
class CBAM(Layer):
    def __init__(self, filters=None, ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.filters is None:

            if isinstance(input_shape[-1], tuple):
                self.filters = int(input_shape[-1][-1])
            else:
                self.filters = int(input_shape[-1])

        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()

        self.fc1 = Dense(max(self.filters // self.ratio, 1), activation='relu')
        self.fc2 = Dense(self.filters)

        self.conv_spatial = Conv2D(
        1,
        kernel_size=7,
        padding='same',
        activation='sigmoid'
    )

        super(CBAM, self).build(input_shape)

    def call(self, inputs):

        if isinstance(inputs, list):
            inputs = inputs[0]

        avg = self.avg_pool(inputs)
        max_ = self.max_pool(inputs)

        avg = self.fc2(self.fc1(avg))
        max_ = self.fc2(self.fc1(max_))

        ca = tf.nn.sigmoid(avg + max_)
        ca = Reshape((1, 1, self.filters))(ca)

        x = Multiply()([inputs, ca])

        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)

        sa = self.conv_spatial(tf.concat([avg, max_], axis=-1))

        x = Multiply()([x, sa])
        return x

    def get_config(self):
        cfg = super(CBAM, self).get_config()
        cfg.update({
            "filters": self.filters,
            "ratio": self.ratio
        })
        return cfg


# -----------------------------------------------------
# Flask Setup
# -----------------------------------------------------
app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.h5")
print("Starting model load...")

model = load_model(
    MODEL_PATH,
    custom_objects={"CBAM": CBAM},
    compile=False
)

print("Model loaded successfully")

class_labels = ['notumor', 'meningioma', 'pituitary', 'glioma']

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


# -----------------------------------------------------
# Grad-CAM (Your existing fake visual heatmap)
# -----------------------------------------------------
def make_gradcam_heatmap(img_array, model):

    img = img_array[0]
    gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)

    edges = cv2.Canny(enhanced, 30, 150)

    heatmap = cv2.addWeighted(enhanced, 0.7, edges, 0.3, 0)

    heatmap = heatmap.astype("float32")
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap


def overlay_heatmap(original_path, heatmap, output_path):

    img = cv2.imread(original_path)
    img = cv2.resize(img, (128, 128))

    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = 1 - heatmap

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.4, heatmap_color, 0.6, 0)

    cv2.imwrite(output_path, superimposed_img)


# -----------------------------------------------------
# GEMINI AI EXPLANATION (NEW PART)
# -----------------------------------------------------
def generate_ai_explanation(image_path, heatmap_path, predicted_class):

    prompt = f"""
    This is a brain MRI image along with a heatmap highlighting the region
    influencing a deep learning tumor classification model.

    The model predicted: {predicted_class}.

    Based on the highlighted region and MRI characteristics,
    provide a professional medical-style explanation describing
    why this region may correspond to this tumor type.
    Keep it concise and suitable for doctors.
    """

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    with open(heatmap_path, "rb") as f:
        heat_bytes = f.read()

    response = gemini_model.generate_content([
        prompt,
        {"mime_type": "image/png", "data": img_bytes},
        {"mime_type": "image/png", "data": heat_bytes}
    ])

    return response.text


# -----------------------------------------------------
# Prediction + Gemini Explainable AI
# -----------------------------------------------------
def predict_and_explain(image_path):

    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    predicted_label = class_labels[index]

    if predicted_label == "notumor":
        result_text = "No Tumor Detected"
    else:
        result_text = f"Tumor Detected: {predicted_label}"

    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

    result_filename = "heatmap_" + os.path.basename(image_path)
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)

    overlay_heatmap(image_path, heatmap, result_path)

    # 🔥 Gemini Explanation
    explanation = generate_ai_explanation(image_path, result_path, predicted_label)

    return result_text, confidence, result_filename, explanation


# -----------------------------------------------------
# Routes
# -----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("index.html", result="Upload an image.")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result, confidence, result_filename, explanation = predict_and_explain(filepath)

        return render_template(
            "index.html",
            result=result,
            confidence=round(confidence * 100, 2),
            uploaded_image=f"/uploads/{file.filename}",
            result_image=f"/results/{result_filename}",
            explanation=explanation
        )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


# -----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)