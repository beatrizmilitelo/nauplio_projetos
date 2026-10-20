import streamlit as st
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Configuração da página
st.set_page_config(page_title="Detector Universal por IA", page_icon="🤖", layout="centered")

st.title("Detector Universal de Imagens com IA 🤖")
st.write("Digite o que quiser em português, aperte **Enter** e faça o upload da foto para testar!")

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
# Função de Tradução Direta (PT -> EN) Sem Lista Fixa
# ---------------------------------------------------------
def traduzir_pt_para_en(texto: str) -> str:
    texto_limpo = texto.strip().lower()
    if not texto_limpo:
        return ""
    try:
        # Força a tradução estritamente do português para o inglês
        traducao = GoogleTranslator(source='pt', target='en').translate(texto_limpo)
        if traducao:
            traducao_limpa = traducao.lower().strip()
            # Se por acaso a API devolver a mesma palavra (como acontece com 'gato'), 
            # tentamos forçar uma tradução geral sem restrição de origem
            if traducao_limpa == texto_limpo:
                traducao_geral = GoogleTranslator(source='auto', target='en').translate(texto_limpo)
                if traducao_geral and traducao_geral.lower().strip() != texto_limpo:
                    return traducao_geral.lower().strip()
            return traducao_limpa
    except Exception:
        pass
    return texto_limpo

def traduzir_para_portugues(texto: str) -> str:
    try:
        return GoogleTranslator(source='en', target='pt').translate(texto).lower().strip()
    except Exception:
        return texto

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
alvo_usuario = st.text_input("O que quer que a IA reconheça?", placeholder="Ex: gato, porco, abacaxi, violão...")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem Enviada", use_column_width=True)
    
    if not alvo_usuario:
        st.warning("⚠️ Por favor, digite o que quer que a IA reconheça no campo acima e pressione Enter.")
    else:
        # Traduz automaticamente do português para o inglês sem lista
        termo_ingles = traduzir_pt_para_en(alvo_usuario)
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
            label_pt = traduzir_para_portugues(label_lower)
            previsoes_traduzidas.append((label_lower, label_pt, prob))

            # Verifica correspondências cruzadas (inglês ou português)
            if (termo_ingles in label_lower) or (label_lower in termo_ingles) or (alvo_lower in label_pt):
                encontrou = True

        if encontrou:
            st.success(f"🤖 Palpite da IA: **Encontrei '{alvo_usuario}' (traduzido para '{termo_ingles}') na imagem!** ✅")
        else:
            st.error(f"🤖 Palpite da IA: **Não consegui identificar '{alvo_usuario}' com segurança.** ❌")

        with st.expander("Ver o que a IA enxergou (Top 5 previsões)"):
            for i, (label_en, label_pt, prob) in enumerate(previsoes_traduzidas):
                st.write(f"{i+1}. **{label_en}** (*{label_pt}*) — Confiança: {prob * 100:.2f}%")

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
