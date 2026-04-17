import io
import glob
from datetime import datetime

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

IMG_HEIGHT, IMG_WIDTH = 150, 150
LAST_CONV_LAYER_NAME = "conv2d_2"
MODEL_PATH = "ia_lichia/modelo_lichia.keras"
LOGO_PATH = "LOGO SPLASH SCREEN.png"

st.set_page_config(
    page_title="Mobo - Análise de Lichias",
    page_icon="🍒",
    layout="wide"
)

# -----------------------------
# ESTILO
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #09111f 0%, #0c1323 100%);
    }

    .hero-box {
        background: linear-gradient(135deg, rgba(108,129,64,0.30), rgba(191,7,90,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.4rem;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.08rem;
        color: #d1d5db;
        margin-top: 0.2rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.7rem;
        margin-bottom: 0.9rem;
    }

    .result-card {
        padding: 1.25rem;
        border-radius: 18px;
        margin-top: 0.8rem;
        margin-bottom: 1.1rem;
        border-left: 8px solid;
        box-shadow: 0 8px 28px rgba(0,0,0,0.18);
    }

    .result-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .result-text {
        font-size: 1rem;
        line-height: 1.6;
    }

    .metric-box {
        background: rgba(10, 24, 52, 0.82);
        padding: 1rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.09);
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .history-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 0.8rem;
        margin-bottom: 0.7rem;
    }

    .small-muted {
        color: #cbd5e1;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# ESTADO
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "selected_example" not in st.session_state:
    st.session_state.selected_example = None


# -----------------------------
# FUNÇÕES
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def get_example_images():
    madura = sorted(glob.glob("ia_lichia/dataset/test/madura/*"))
    nao_madura = sorted(glob.glob("ia_lichia/dataset/test/nao_madura/*"))

    examples = {}
    if madura:
        examples["Exemplo madura"] = madura[0]
    if nao_madura:
        examples["Exemplo não madura"] = nao_madura[0]

    return examples


def preprocess_image(image):
    image = image.convert("RGB")
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_resized, image_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.45):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize(original_image.size)

    original_array = np.array(original_image, dtype=np.float32)
    heatmap_array = np.array(jet_heatmap, dtype=np.float32)

    superimposed = heatmap_array * alpha + original_array
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)


def get_result_style(confidence):
    if confidence >= 0.90:
        return {
            "bg": "#08351d",
            "border": "#22c55e",
            "title": "Alta confiança",
            "text": "O modelo identificou o resultado com alta segurança."
        }
    elif confidence >= 0.70:
        return {
            "bg": "#3e2c0d",
            "border": "#facc15",
            "title": "Confiança moderada",
            "text": "O modelo encontrou um resultado plausível, mas com segurança intermediária."
        }
    else:
        return {
            "bg": "#3b1115",
            "border": "#ef4444",
            "title": "Baixa confiança",
            "text": "O modelo encontrou um resultado incerto. Vale a pena testar outra imagem."
        }


def get_prediction_text(predicted_class, confidence):
    if predicted_class == "madura":
        return f"A IA identificou que a lichia está madura com confiança de {confidence:.2%}."
    return f"A IA identificou que a lichia está não madura com confiança de {confidence:.2%}."


def safe_font(size=28):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_report_image(original_image, heatmap_image, predicted_class, confidence, timestamp_text):
    width = 1600
    height = 980
    bg = (11, 18, 35)
    accent = (191, 7, 90)
    green = (34, 197, 94)
    white = (245, 245, 245)
    soft = (203, 213, 225)

    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    title_font = safe_font(54)
    subtitle_font = safe_font(28)
    text_font = safe_font(34)
    small_font = safe_font(24)

    # Cabeçalho
    draw.rounded_rectangle((40, 35, 1560, 170), radius=30, fill=(108, 129, 64))
    draw.text((70, 60), "Relatório de análise - Mobo IA Lichia", fill=white, font=title_font)
    draw.text((70, 122), f"Gerado em: {timestamp_text}", fill=white, font=subtitle_font)

    # Cartão de resultado
    result_color = green if confidence >= 0.90 else accent
    draw.rounded_rectangle((40, 205, 1560, 355), radius=26, fill=(15, 41, 31))
    draw.text((70, 235), f"Classe prevista: {predicted_class}", fill=white, font=text_font)
    draw.text((70, 285), f"Confiança: {confidence:.2%}", fill=result_color, font=text_font)

    # Imagens
    img_w, img_h = 650, 520
    original_resized = original_image.resize((img_w, img_h))
    heatmap_resized = heatmap_image.resize((img_w, img_h))

    canvas.paste(original_resized, (70, 400))
    canvas.paste(heatmap_resized, (880, 400))

    draw.rounded_rectangle((60, 390, 730, 930), radius=18, outline=white, width=3)
    draw.rounded_rectangle((870, 390, 1540, 930), radius=18, outline=white, width=3)

    draw.text((280, 935), "Imagem original", fill=soft, font=small_font)
    draw.text((1060, 935), "Heatmap da atenção da IA", fill=soft, font=small_font)

    return canvas


def pil_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def pil_to_pdf_bytes(image):
    buffer = io.BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffer, format="PDF")
    buffer.seek(0)
    return buffer


def add_to_history(file_name, predicted_class, confidence):
    entry = {
        "file_name": file_name,
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.2%}",
        "time": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]


# -----------------------------
# CARREGAMENTO
# -----------------------------
model = load_model()
examples = get_example_images()


# -----------------------------
# SIDEBAR / HISTÓRICO
# -----------------------------
with st.sidebar:
    st.markdown("## Histórico de previsões")
    if st.session_state.history:
        for item in st.session_state.history:
            st.markdown(
                f"""
                <div class="history-card">
                    <strong>{item['predicted_class']}</strong><br>
                    <span class="small-muted">{item['file_name']}</span><br>
                    <span class="small-muted">Confiança: {item['confidence']}</span><br>
                    <span class="small-muted">{item['time']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Nenhuma previsão registrada ainda.")

    if st.button("Limpar histórico", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# -----------------------------
# TOPO
# -----------------------------
top_col1, top_col2 = st.columns([1, 6])

with top_col1:
    try:
        st.image(LOGO_PATH, width=110)
    except Exception:
        st.markdown("### 🍒")

with top_col2:
    st.markdown(
        """
        <div class="hero-box">
            <div class="main-title">IA para análise de maturação de lichias</div>
            <div class="subtitle">
                Envie uma imagem para que o modelo identifique se a lichia está madura ou não madura
                e mostre a região visual mais relevante para a decisão.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# EXEMPLOS
# -----------------------------
st.markdown('<div class="section-title">Teste rápido</div>', unsafe_allow_html=True)
example_cols = st.columns(3)

with example_cols[0]:
    if "Exemplo madura" in examples and st.button("Usar exemplo madura", use_container_width=True):
        st.session_state.selected_example = examples["Exemplo madura"]

with example_cols[1]:
    if "Exemplo não madura" in examples and st.button("Usar exemplo não madura", use_container_width=True):
        st.session_state.selected_example = examples["Exemplo não madura"]

with example_cols[2]:
    if st.button("Remover exemplo selecionado", use_container_width=True):
        st.session_state.selected_example = None

uploaded_file = st.file_uploader(
    "Escolha uma imagem",
    type=["jpg", "jpeg", "png"]
)

image_source_name = None
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_source_name = uploaded_file.name
elif st.session_state.selected_example is not None:
    image = Image.open(st.session_state.selected_example).convert("RGB")
    image_source_name = st.session_state.selected_example.split("\\")[-1].split("/")[-1]


# -----------------------------
# ANÁLISE
# -----------------------------
if image is not None:
    processed_image, image_array = preprocess_image(image)

    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        predicted_class = "não madura"
        confidence = prediction
    else:
        predicted_class = "madura"
        confidence = 1 - prediction

    heatmap = make_gradcam_heatmap(image_array, model, LAST_CONV_LAYER_NAME)
    heatmap_image = overlay_heatmap_on_image(processed_image, heatmap)

    style = get_result_style(confidence)
    prediction_text = get_prediction_text(predicted_class, confidence)

    add_to_history(image_source_name, predicted_class, confidence)

    st.markdown('<div class="section-title">Resultado da análise</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card" style="background-color:{style['bg']}; border-left-color:{style['border']};">
            <div class="result-title" style="color:{style['border']};">{style['title']}</div>
            <div class="result-text">
                <strong>Classe prevista:</strong> {predicted_class}<br>
                <strong>Confiança:</strong> {confidence:.2%}<br><br>
                {prediction_text}<br>
                {style['text']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_metric1, col_metric2 = st.columns(2)

    with col_metric1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Classe prevista</div>
                <div class="metric-value">{predicted_class}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_metric2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Confiança</div>
                <div class="metric-value">{confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(float(confidence))

    st.markdown('<div class="section-title">Visualização</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.image(processed_image, caption="Imagem original", use_container_width=True)

    with col2:
        st.image(heatmap_image, caption="Heatmap da atenção da IA", use_container_width=True)

    with st.expander("Como interpretar o heatmap"):
        st.write(
            "O heatmap destaca as regiões da imagem que mais influenciaram a decisão do modelo. "
            "Áreas mais quentes tendem a indicar onde a rede concentrou mais atenção para classificar a lichia."
        )

    # Exportação
    st.markdown('<div class="section-title">Exportação</div>', unsafe_allow_html=True)

    report_image = make_report_image(
        original_image=processed_image,
        heatmap_image=heatmap_image,
        predicted_class=predicted_class,
        confidence=confidence,
        timestamp_text=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )

    png_buffer = pil_to_png_bytes(report_image)
    pdf_buffer = pil_to_pdf_bytes(report_image)

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.download_button(
            label="Baixar análise em PNG",
            data=png_buffer,
            file_name="analise_lichia.png",
            mime="image/png",
            use_container_width=True
        )

    with export_col2:
        st.download_button(
            label="Baixar análise em PDF",
            data=pdf_buffer,
            file_name="analise_lichia.pdf",
            mime="application/pdf",
            use_container_width=True
        )