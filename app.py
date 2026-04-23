import io
import glob
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# CONFIGURAÇÕES GERAIS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
IMG_HEIGHT, IMG_WIDTH = 150, 150
LAST_CONV_LAYER_NAME = "conv2d_2"

MODEL_PATH = BASE_DIR / "ia_lichia" / "modelo_lichia.keras"
LOGO_PATH = BASE_DIR / "LOGO SPLASH SCREEN.png"
CSS_PATH = BASE_DIR / "styles.css"


# -----------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------
st.set_page_config(
    page_title="Mobo - Análise de Lichias",
    page_icon="🍒",
    layout="wide"
)


# -----------------------------
# ESTILOS
# -----------------------------
def load_css():
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------
# ESTADO
# -----------------------------
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "selected_example" not in st.session_state:
        st.session_state.selected_example = None

    if "selected_analysis_index" not in st.session_state:
        st.session_state.selected_analysis_index = 0

    if "last_history_key" not in st.session_state:
        st.session_state.last_history_key = None


# -----------------------------
# CARREGAMENTO DE RECURSOS
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def get_example_images():
    madura = sorted(
        glob.glob(str(BASE_DIR / "ia_lichia" / "dataset" / "test" / "madura" / "*"))
    )
    nao_madura = sorted(
        glob.glob(str(BASE_DIR / "ia_lichia" / "dataset" / "test" / "nao_madura" / "*"))
    )

    examples = {}
    if madura:
        examples["Exemplo madura"] = madura[0]
    if nao_madura:
        examples["Exemplo não madura"] = nao_madura[0]

    return examples


# -----------------------------
# PROCESSAMENTO DE IMAGEM
# -----------------------------
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

    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize(original_image.size)

    original_array = np.array(original_image, dtype=np.float32)
    heatmap_array = np.array(jet_heatmap, dtype=np.float32)

    superimposed = heatmap_array * alpha + original_array
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)


# -----------------------------
# REGRAS DE APRESENTAÇÃO
# -----------------------------
def get_result_style(confidence):
    if confidence >= 0.90:
        return {
            "bg": "#08351d",
            "border": "#22c55e",
            "title": "Alta confiança",
            "text": "O modelo identificou o resultado com alta segurança."
        }
    if confidence >= 0.70:
        return {
            "bg": "#3e2c0d",
            "border": "#facc15",
            "title": "Confiança moderada",
            "text": "O modelo encontrou um resultado plausível, mas com segurança intermediária."
        }
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


# -----------------------------
# EXPORTAÇÃO
# -----------------------------
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

    draw.rounded_rectangle((40, 35, 1560, 170), radius=30, fill=(108, 129, 64))
    draw.text((70, 60), "Relatório de análise - Mobo IA Lichia", fill=white, font=title_font)
    draw.text((70, 122), f"Gerado em: {timestamp_text}", fill=white, font=subtitle_font)

    result_color = green if confidence >= 0.90 else accent
    draw.rounded_rectangle((40, 205, 1560, 355), radius=26, fill=(15, 41, 31))
    draw.text((70, 235), f"Classe prevista: {predicted_class}", fill=white, font=text_font)
    draw.text((70, 285), f"Confiança: {confidence:.2%}", fill=result_color, font=text_font)

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


# -----------------------------
# HISTÓRICO
# -----------------------------
def add_to_history(file_name, predicted_class, confidence):
    entry = {
        "file_name": file_name,
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.2%}",
        "time": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]


def register_history_once(result):
    history_key = (
        result["file_name"],
        result["predicted_class"],
        f"{result['confidence']:.4f}"
    )

    if st.session_state.last_history_key != history_key:
        add_to_history(
            result["file_name"],
            result["predicted_class"],
            result["confidence"]
        )
        st.session_state.last_history_key = history_key


# -----------------------------
# ANÁLISE
# -----------------------------
def analyze_image(image, image_source_name, model):
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

    return {
        "file_name": image_source_name,
        "processed_image": processed_image,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "heatmap_image": heatmap_image,
        "style": get_result_style(confidence),
        "prediction_text": get_prediction_text(predicted_class, confidence)
    }


def build_selected_images(uploaded_files):
    selected_images = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            selected_images.append({
                "name": uploaded_file.name,
                "image": Image.open(uploaded_file).convert("RGB")
            })
    elif st.session_state.selected_example is not None:
        selected_images.append({
            "name": Path(st.session_state.selected_example).name,
            "image": Image.open(st.session_state.selected_example).convert("RGB")
        })

    if not selected_images:
        st.session_state.selected_analysis_index = 0
    elif st.session_state.selected_analysis_index >= len(selected_images):
        st.session_state.selected_analysis_index = 0

    return selected_images


# -----------------------------
# COMPONENTES DE INTERFACE
# -----------------------------
def render_sidebar_history():
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

        if st.button("Limpar histórico", width="stretch"):
            st.session_state.history = []
            st.session_state.last_history_key = None
            st.rerun()


def render_header():
    top_col1, top_col2 = st.columns([1, 6])

    with top_col1:
        try:
            st.image(str(LOGO_PATH), width=110)
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


def render_example_buttons(examples):
    st.markdown('<div class="section-title">Teste rápido</div>', unsafe_allow_html=True)
    example_cols = st.columns(3)

    with example_cols[0]:
        if "Exemplo madura" in examples and st.button("Usar exemplo madura", width="stretch"):
            st.session_state.selected_example = examples["Exemplo madura"]
            st.session_state.selected_analysis_index = 0

    with example_cols[1]:
        if "Exemplo não madura" in examples and st.button("Usar exemplo não madura", width="stretch"):
            st.session_state.selected_example = examples["Exemplo não madura"]
            st.session_state.selected_analysis_index = 0

    with example_cols[2]:
        if st.button("Remover exemplo selecionado", width="stretch"):
            st.session_state.selected_example = None
            st.session_state.selected_analysis_index = 0


def render_comparison_cards(analysis_results):
    st.markdown('<div class="section-title">Comparação das análises</div>', unsafe_allow_html=True)

    num_columns = min(len(analysis_results), 3)
    comparison_cols = st.columns(num_columns)

    for index, result in enumerate(analysis_results):
        col = comparison_cols[index % num_columns]
        with col:
            st.image(result["processed_image"], width="stretch")
            st.markdown(
                f'<div class="comparison-image-caption">{result["file_name"]}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="comparison-card" style="border-left: 6px solid {result['style']['border']};">
                    <div class="comparison-title" style="color:{result['style']['border']};">
                        {result['predicted_class']}
                    </div>
                    <div class="comparison-text">
                        <strong>Confiança:</strong> {result['confidence']:.2%}<br>
                        {result['prediction_text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Ver análise detalhada",
                key=f"select_analysis_{index}",
                width="stretch"
            ):
                st.session_state.selected_analysis_index = index
                st.rerun()

            if st.session_state.selected_analysis_index == index:
                st.markdown(
                    '<div class="selected-chip">Análise selecionada</div>',
                    unsafe_allow_html=True
                )


def render_detailed_result(primary_result):
    st.markdown('<div class="detail-wrapper"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Resultado detalhado</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card" style="background-color:{primary_result['style']['bg']}; border-left-color:{primary_result['style']['border']};">
            <div class="result-title" style="color:{primary_result['style']['border']};">{primary_result['style']['title']}</div>
            <div class="result-text">
                <strong>Arquivo:</strong> {primary_result['file_name']}<br>
                <strong>Classe prevista:</strong> {primary_result['predicted_class']}<br>
                <strong>Confiança:</strong> {primary_result['confidence']:.2%}<br><br>
                {primary_result['prediction_text']}<br>
                {primary_result['style']['text']}
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
                <div class="metric-value">{primary_result['predicted_class']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_metric2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Confiança</div>
                <div class="metric-value">{primary_result['confidence']:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(float(primary_result["confidence"]))


def render_detailed_visualization(primary_result):
    st.markdown('<div class="section-title">Visualização detalhada</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.image(primary_result["processed_image"], caption="Imagem original", width="stretch")

    with col2:
        st.image(primary_result["heatmap_image"], caption="Heatmap da atenção da IA", width="stretch")

    with st.expander("Como interpretar o heatmap"):
        st.write(
            "O heatmap destaca as regiões da imagem que mais influenciaram a decisão do modelo. "
            "Áreas mais quentes tendem a indicar onde a rede concentrou mais atenção para classificar a lichia."
        )


def render_export_section(primary_result):
    st.markdown('<div class="section-title">Exportação</div>', unsafe_allow_html=True)

    report_image = make_report_image(
        original_image=primary_result["processed_image"],
        heatmap_image=primary_result["heatmap_image"],
        predicted_class=primary_result["predicted_class"],
        confidence=primary_result["confidence"],
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
            width="stretch"
        )

    with export_col2:
        st.download_button(
            label="Baixar análise em PDF",
            data=pdf_buffer,
            file_name="analise_lichia.pdf",
            mime="application/pdf",
            width="stretch"
        )


# -----------------------------
# FLUXO PRINCIPAL
# -----------------------------
load_css()
init_session_state()

model = load_model()
examples = get_example_images()

render_sidebar_history()
render_header()
render_example_buttons(examples)

uploaded_files = st.file_uploader(
    "Escolha uma ou mais imagens",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

selected_images = build_selected_images(uploaded_files)

if selected_images:
    analysis_results = [
        analyze_image(item["image"], item["name"], model)
        for item in selected_images
    ]

    render_comparison_cards(analysis_results)

    primary_result = analysis_results[st.session_state.selected_analysis_index]
    register_history_once(primary_result)

    render_detailed_result(primary_result)
    render_detailed_visualization(primary_result)
    render_export_section(primary_result)