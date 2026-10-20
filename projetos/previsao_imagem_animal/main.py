import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Configuração da página
st.set_page_config(page_title="Detector de Porcos + Métricas", page_icon="🐷", layout="centered")

st.title("Detector de Porcos com IA 🐷")
st.write("Faça o upload de uma imagem para verificar se é um porco e acompanhe a taxa de acertos!")

# ---------------------------------------------------------
# Inicialização das variáveis na sessão (st.session_state)
# ---------------------------------------------------------
if 'total_rodadas' not in st.session_state:
    st.session_state.total_rodadas = 0
if 'acertos' not in st.session_state:
    st.session_state.acertos = 0
if 'erros' not in st.session_state:
    st.session_state.erros = 0
if 'ultima_predicao' not in st.session_state:
    st.session_state.ultima_predicao = None
if 'avaliado' not in st.session_state:
    st.session_state.avaliado = True

# Functions para registrar acerto/erro
def registrar_resultado(correto: bool):
    if st.session_state.total_rodadas < 100 and not st.session_state.avaliado:
        st.session_state.total_rodadas += 1
        if correto:
            st.session_state.acertos += 1
        else:
            st.session_state.erros += 1
        st.session_state.avaliado = True

def resetar_metricas():
    st.session_state.total_rodadas = 0
    st.session_state.acertos = 0
    st.session_state.erros = 0
    st.session_state.ultima_predicao = None
    st.session_state.avaliado = True

# ---------------------------------------------------------
# Carregar o modelo de IA
# ---------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    return MobileNetV2(weights='imagenet')

with st.spinner("Carregando o modelo de Inteligência Artificial..."):
    model = carregar_modelo()

# ---------------------------------------------------------
# Exibição do Painel de Métricas (100 Rodadas)
# ---------------------------------------------------------
st.subheader("📊 Painel de Desempenho (Meta: 100 Rodadas)")

total = st.session_state.total_rodadas
acertos = st.session_state.acertos
erros = st.session_state.erros

pct_acertos = (acertos / total * 100) if total > 0 else 0.0
pct_erros = (erros / total * 100) if total > 0 else 0.0

# Colunas para visualização visual dos dados
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rodadas", f"{total} / 100")
col2.metric("Acertos", f"{acertos}", f"{pct_acertos:.1f}%")
col3.metric("Erros", f"{erros}", f"{pct_erros:.1f}%", delta_color="inverse")
col4.metric("Acurácia Total", f"{pct_acertos:.1f}%")

# Barra de progresso das 100 rodadas
st.progress(min(total / 100, 1.0))

if total >= 100:
    st.warning("⚠️ Você atingiu o limite de 100 rodadas!")

if st.button("🔄 Resetar Métricas"):
    resetar_metricas()
    st.rerun()

st.divider()

# ---------------------------------------------------------
# Upload e Predição da Imagem
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem Enviada", use_column_width=True)
    
    # Processar apenas se for uma nova imagem ou se ainda não foi feito
    img_resized = img.resize((224, 224))
    x = keras_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    
    is_pig = False
    for _, label, prob in decoded:
        if any(termo in label.lower() for termo in ['pig', 'hog', 'boar', 'swine']):
            is_pig = True
            break

    # Exibir o palpite da IA
    if is_pig:
        st.success("🤖 Palpite da IA: **É um Porco!** 🐷")
    else:
        st.error("🤖 Palpite da IA: **NÃO é um Porco.** ❌")

    # Detalhes técnicos da IA
    with st.expander("Ver detalhes do modelo"):
        for i, (_, label, prob) in enumerate(decoded):
            st.write(f"{i+1}. **{label}** ({prob * 100:.2f}%)")

    st.divider()

    # ---------------------------------------------------------
    # Avaliação do Usuário para Contabilizar nas Métricas
    # ---------------------------------------------------------
    if total < 100:
        st.write("**A IA acertou o palpite desta imagem?**")
        col_sim, col_nao = st.columns(2)
        
        with col_sim:
            if st.button("✅ Sim, a IA acertou", use_container_width=True):
                registrar_resultado(correto=True)
                st.rerun()
                
        with col_nao:
            if st.button("❌ Não, a IA errou", use_container_width=True):
                registrar_resultado(correto=False)
                st.rerun()
