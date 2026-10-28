import streamlit as st
import numpy as np
import urllib.parse
import urllib.request
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Configuração da página
st.set_page_config(page_title="Detector Universal por IA", page_icon="🤖", layout="centered")

st.title("Detector Universal de Imagens com IA 🤖")
st.write("Digite o que quiser em **português** (ex: *gato, cachorro, violão*), aperte **Enter** e envie a foto!")

# ---------------------------------------------------------
# Inicialização das variáveis na sessão (st.session_state)
# ---------------------------------------------------------
if 'total_rodadas' not in st.session_state:
    st.session_state.total_rodadas = 0
if 'acertos' not in st.session_state:
    st.session_state.acertos = 0
if 'erros' not in st.session_state:
    st.session_state.erros = 0

def registrar_resultado(correto: bool):
    if st.session_state.total_rodadas < 100:
        st.session_state.total_rodadas += 1
        if correto:
            st.session_state.acertos += 1
        else:
            st.session_state.erros += 1

def resetar_metricas():
    st.session_state.total_rodadas = 0
    st.session_state.acertos = 0
    st.session_state.erros = 0

# ---------------------------------------------------------
# Tradutor Robusto via API Pública do Google (PT -> EN / EN -> PT)
# ---------------------------------------------------------
def traduzir_texto(texto: str, origem: str, destino: str) -> str:
    texto_limpo = texto.strip().lower()
    if not texto_limpo:
        return ""
    try:
        encoded_text = urllib.parse.quote(texto_limpo)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={origem}&tl={destino}&dt=t&q={encoded_text}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            traduzido = result[0][0][0]
            if traduzido:
                return traduzido.lower().strip()
    except Exception:
        pass
    return texto_limpo

# ---------------------------------------------------------
# Carregar o modelo de IA
# ---------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    return MobileNetV2(weights='imagenet')

with st.spinner("A carregar o modelo de Inteligência Artificial..."):
    model = carregar_modelo()

# ---------------------------------------------------------
# Painel de Métricas (100 Rodadas)
# ---------------------------------------------------------
st.subheader("📊 Painel de Desempenho (Meta: 100 Rodadas)")

total = st.session_state.total_rodadas
acertos = st.session_state.acertos
erros = st.session_state.erros

pct_acertos = (acertos / total * 100) if total > 0 else 0.0
pct_erros = (erros / total * 100) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rodadas", f"{total} / 100")
col2.metric("Acertos", f"{acertos}", f"{pct_acertos:.1f}%")
col3.metric("Erros", f"{erros}", f"{pct_erros:.1f}%", delta_color="inverse")
col4.metric("Acurácia Total", f"{pct_acertos:.1f}%")

st.progress(min(total / 100, 1.0))

if total >= 100:
    st.warning("⚠️ Atingiu o limite de 100 rodadas!")

if st.button("🔄 Resetar Métricas"):
    resetar_metricas()
    st.rerun()

st.divider()

# ---------------------------------------------------------
# Entrada do Alvo e Upload da Imagem
# ---------------------------------------------------------
alvo_usuario = st.text_input("O que quer que a IA reconheça?", placeholder="Ex: gato, cachorro, abacaxi, violão...")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem Enviada", use_column_width=True)
    
    if not alvo_usuario:
        st.warning("⚠️ Por favor, digite o que quer que a IA reconheça no campo acima e pressione Enter.")
    else:
        # Traduz rigorosamente do Português (pt) para o Inglês (en)
        termo_ingles = traduzir_texto(alvo_usuario, 'pt', 'en')
        alvo_lower = alvo_usuario.lower().strip()
        
        st.write(f"Analisando a imagem procurando por: *{alvo_usuario}* (Traduzido para inglês: **{termo_ingles}**)...")
        
        img_resized = img.resize((224, 224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        decoded = decode_predictions(preds, top=5)[0]
        
        encontrou = False
        previsoes_traduzidas = []

        for _, label, prob in decoded:
            label_lower = label.lower().replace('_', ' ')
            # Traduz a classe do banco da IA do Inglês para o Português
            label_pt = traduzir_texto(label_lower, 'en', 'pt')
            previsoes_traduzidas.append((label_lower, label_pt, prob))

            # Compara tanto o termo em inglês traduzido quanto o termo em português digitado
            if (termo_ingles in label_lower) or (label_lower in termo_ingles) or \
               (alvo_lower in label_pt) or (label_pt in alvo_lower):
                encontrou = True

        if encontrou:
            st.success(f"🤖 Palpite da IA: **Encontrei '{alvo_usuario}' (detectado como '{termo_ingles}') na imagem!** ✅")
        else:
            st.error(f"🤖 Palpite da IA: **Não consegui identificar '{alvo_usuario}' com segurança.** ❌")

        with st.expander("Ver o que a IA enxergou (Top 5 previsões traduzidas)"):
            for i, (label_en, label_pt, prob) in enumerate(previsoes_traduzidas):
                st.write(f"{i+1}. **{label_pt.title()}** (*{label_en}*) — Confiança: {prob * 100:.2f}%")

        st.divider()

        # ---------------------------------------------------------
        # Validação para as métricas
        # ---------------------------------------------------------
        if total < 100:
            st.write("**A IA identificou corretamente o seu pedido?**")
            col_sim, col_nao = st.columns(2)
            
            with col_sim:
                if st.button("✅ Sim, acertou", use_container_width=True):
                    registrar_resultado(correto=True)
                    st.rerun()
                    
            with col_nao:
                if st.button("❌ Não, errou", use_container_width=True):
                    registrar_resultado(correto=False)
                    st.rerun()
