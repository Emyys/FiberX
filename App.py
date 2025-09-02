import streamlit as st
import opencv-python
from ultralytics import YOLO
from PIL import Image
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    
    page_bg = f"""
    <style>
    /* Mantém a imagem de fundo sem sobreposição */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Título com texto claro */
    h1 {{
        color: white !important;
    }}

    /* Barra lateral com fundo branco e texto escuro */
    section[data-testid="stSidebar"] {{
        background-color: white;
        color: black;
    }}
    section[data-testid="stSidebar"] * {{
        color: black !important;
    }}

    /* Textos gerais escuros */
    h2, h3, h4, h5, h6, p, div, label, span {{
        color: black !important;
    }}

    /* Inputs e botões com fundo branco e texto escuro */
    input, textarea, select, .stButton>button, .stSlider>div {{
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 5px !important;
    }}

    .stButton>button:hover {{
        background-color: #f0f0f0 !important;
    }}

    /* Área de upload com fundo branco e texto escuro */
    div[data-testid="stFileUploadDropzone"] {{
        background-color: white !important;
        border: 2px dashed #999 !important;
        color: black !important;
        padding: 20px;
        border-radius: 10px;
    }}
    .stFileUploader label, .stFileUploader span, .stFileUploader div {{
        color: black !important;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)


def main():
    set_background(r"C:\Users\emilimoraes\fundo.png")

    st.title("Classificação De Sujidade")

    model = YOLO("my_model.pt")

    option = st.sidebar.radio(
        "Escolha a fonte de entrada:",
        ("Câmera", "Vídeo", "Imagem")
    )

    if option == "Câmera":
        if st.button("Iniciar câmera"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            stop_btn = st.button("Parar câmera")

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.write("Não foi possível acessar a câmera")
                    break

                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR")

            cap.release()

    elif option == "Vídeo":
        uploaded_file = st.file_uploader("Envie um vídeo", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR")

            cap.release()

    elif option == "Imagem":
        uploaded_image = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            results = model(image, verbose=False)
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="Resultado YOLO", channels="BGR")


if __name__ == "__main__":
    main()

