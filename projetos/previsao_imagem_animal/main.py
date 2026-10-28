import streamlit as st
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------
st.set_page_config(page_title="Detector Universal por IA", page_icon="🤖", layout="centered")
st.title("Detector Universal de Imagens com IA 🤖")
st.write("Digite **qualquer coisa em português** (ex: *gato, violão, abacaxi, bicicleta vermelha*), envie a foto e veja se a IA encontra.")

# ---------------------------------------------------------
# Sessão e métricas
# ---------------------------------------------------------
for chave in ("total_rodadas", "acertos", "erros"):
    if chave not in st.session_state:
        st.session_state[chave] = 0

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
# Modelos CLIP (carregados uma vez)
# ---------------------------------------------------------
@st.cache_resource
def carregar_modelos():
    modelo_imagem = SentenceTransformer("clip-ViT-B-32")  # codifica imagens
    modelo_texto = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")  # codifica texto em PT/EN/etc.
    return modelo_imagem, modelo_texto

with st.spinner("Carregando o modelo CLIP (na primeira vez pode demorar)..."):
    modelo_imagem, modelo_texto = carregar_modelos()

# Termos de contraste em português: o alvo precisa ganhar deles
TERMOS_CONTRASTE = [
    "uma pessoa", "um animal", "um cachorro", "um gato", "uma paisagem",
    "um prédio", "um carro", "comida", "um objeto", "uma planta",
    "um pássaro", "uma flor", "um móvel", "um eletrônico", "uma imagem qualquer",
]

@st.cache_data(show_spinner=False)
def embedding_texto(termos: tuple):
    return modelo_texto.encode(list(termos), convert_to_tensor=True)

def analisar(img: Image.Image, termo: str):
    """Retorna a probabilidade (0-1) de o termo estar na imagem, frente aos termos de contraste."""
    termos = tuple([termo] + TERMOS_CONTRASTE)
    emb_txt = embedding_texto(termos)
    emb_img = modelo_imagem.encode(img, convert_to_tensor=True)
    similaridades = util.cos_sim(emb_img, emb_txt)[0].cpu().numpy()
    
    # Softmax com temperatura de 100 (escala usada pelo CLIP)
    logits = 100.0 * similaridades
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    return float(probs[0]), termos, probs

# ---------------------------------------------------------
# Painel de métricas
# ---------------------------------------------------------
st.subheader("📊 Painel de Desempenho (Meta: 100 Rodadas)")
total = st.session_state.total_rodadas
acertos = st.session_state.acertos
erros = st.session_state.erros
pct_acertos = (acertos / total * 100) if total > 0 else 0.0
pct_erros = (erros / total * 100) if total > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rodadas", f"{total} / 100")
c2.metric("Acertos", f"{acertos}", f"{pct_acertos:.1f}%")
c3.metric("Erros", f"{erros}", f"{pct_erros:.1f}%", delta_color="inverse")
c4.metric("Acurácia Total", f"{pct_acertos:.1f}%")
st.progress(min(total / 100, 1.0))

if total >= 100:
    st.warning("⚠️ Atingiu o limite de 100 rodadas!")
if st.button("🔄 Resetar Métricas"):
    resetar_metricas()
    st.rerun()
st.divider()

# ---------------------------------------------------------
# Entrada
# ---------------------------------------------------------
limite = st.slider("Limite de confiança para dizer 'encontrei'", 0.10, 0.90, 0.50, 0.05)
alvo_usuario = st.text_input("O que quer que a IA reconheça?", placeholder="Ex: gato, violão, abacaxi...")
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem Enviada", use_container_width=True)

    if not alvo_usuario.strip():
        st.warning("⚠️ Digite o que quer que a IA reconheça no campo acima e pressione Enter.")
    else:
        prob_alvo, termos, probs = analisar(img, alvo_usuario.strip())
        encontrou = prob_alvo >= limite

        st.write(f"Confiança para **{alvo_usuario}**: **{prob_alvo * 100:.1f}%** (limite: {limite * 100:.0f}%)")

        if encontrou:
            st.success(f"🤖 Palpite da IA: **Encontrei '{alvo_usuario}' na imagem!** ✅")
        else:
            st.error(f"🤖 Palpite da IA: **Não consegui identificar '{alvo_usuario}' com segurança.** ❌")

        with st.expander("Ver a comparação com os outros termos"):
            ranking = sorted(zip(termos, probs), key=lambda t: t[1], reverse=True)
            for i, (termo, p) in enumerate(ranking, 1):
                st.write(f"{i}. **{termo}** — {p * 100:.2f}%")

        st.divider()

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
