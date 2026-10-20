
import json
import os
import re
import unicodedata

import streamlit as st
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image

# ---------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------
st.set_page_config(
    page_title="Detector Universal por IA",
    page_icon="🤖",
    layout="centered",
)

ARQUIVO_FEEDBACK = "feedback_ia.json"
LIMITE_RODADAS = 100

st.title("Detector Universal de Imagens com IA 🤖")
st.write(
    "Escreva o que deseja identificar em qualquer idioma, "
    "envie uma imagem e avalie o resultado para ajudar o "
    "sistema a melhorar."
)

# ---------------------------------------------------------
# ESTADO DA SESSÃO
# ---------------------------------------------------------
if "total_rodadas" not in st.session_state:
    st.session_state.total_rodadas = 0
if "acertos" not in st.session_state:
    st.session_state.acertos = 0
if "erros" not in st.session_state:
    st.session_state.erros = 0
if "ultima_avaliacao" not in st.session_state:
    st.session_state.ultima_avaliacao = None


# ---------------------------------------------------------
# NORMALIZAÇÃO DOS TERMOS
# ---------------------------------------------------------
def normalizar(texto):
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )
    texto = texto.replace("_", " ")
    texto = re.sub(r"[^\w\s-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


# ---------------------------------------------------------
# TRADUÇÃO AUTOMÁTICA PARA INGLÊS
# ---------------------------------------------------------
def traduzir_para_ingles(texto):
    texto = texto.strip()

    if not texto:
        raise ValueError("Digite um termo para identificar.")

    erros = []

    # Primeiro tenta detectar automaticamente o idioma.
    try:
        resultado = GoogleTranslator(
            source="auto",
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip().lower()

    except Exception as erro:
        erros.append(str(erro))

    # Se falhar, tenta português -> inglês.
    try:
        resultado = GoogleTranslator(
            source="pt",
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip().lower()

    except Exception as erro:
        erros.append(str(erro))

    raise RuntimeError(
        "Não foi possível traduzir o termo. "
        "Verifique sua conexão com a internet e tente novamente."
    )


# ---------------------------------------------------------
# TRADUÇÃO DAS PREVISÕES PARA PORTUGUÊS
# Apenas para exibição ao usuário.
# ---------------------------------------------------------
def traduzir_para_portugues(texto):
    try:
        return GoogleTranslator(
            source="en",
            target="pt",
        ).translate(texto).lower().strip()
    except Exception:
        return texto


# ---------------------------------------------------------
# CARREGAR MODELO
# ---------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    return MobileNetV2(weights="imagenet")


with st.spinner("Carregando o modelo de inteligência artificial..."):
    model = carregar_modelo()


# ---------------------------------------------------------
# CARREGAR E SALVAR FEEDBACK
# ---------------------------------------------------------
def carregar_feedback():
    if not os.path.exists(ARQUIVO_FEEDBACK):
        return {}

    try:
        with open(ARQUIVO_FEEDBACK, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        if isinstance(dados, dict):
            return dados

    except (json.JSONDecodeError, OSError):
        st.warning(
            "O histórico de aprendizado não pôde ser lido. "
            "Um novo histórico será iniciado."
        )

    return {}


def salvar_feedback(feedback):
    # Escrita temporária para reduzir o risco de corromper o arquivo.
    temporario = ARQUIVO_FEEDBACK + ".tmp"

    with open(temporario, "w", encoding="utf-8") as arquivo:
        json.dump(
            feedback,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temporario, ARQUIVO_FEEDBACK)


# ---------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------
def registrar_resultado(correto):
    if st.session_state.total_rodadas >= LIMITE_RODADAS:
        return

    st.session_state.total_rodadas += 1

    if correto:
        st.session_state.acertos += 1
    else:
        st.session_state.erros += 1


def resetar_metricas():
    st.session_state.total_rodadas = 0
    st.session_state.acertos = 0
    st.session_state.erros = 0
    st.session_state.ultima_avaliacao = None


# ---------------------------------------------------------
# PAINEL
# ---------------------------------------------------------
st.subheader("📊 Painel de desempenho")

total = st.session_state.total_rodadas
acertos = st.session_state.acertos
erros = st.session_state.erros

pct_acertos = acertos / total * 100 if total else 0
pct_erros = erros / total * 100 if total else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rodadas", f"{total}/{LIMITE_RODADAS}")
col2.metric("Acertos", acertos, f"{pct_acertos:.1f}%")
col3.metric("Erros", erros, f"{pct_erros:.1f}%", delta_color="inverse")
col4.metric("Acurácia", f"{pct_acertos:.1f}%")

st.progress(min(total / LIMITE_RODADAS, 1.0))

feedback = carregar_feedback()

st.caption(
    f"Termos com histórico de feedback: {len(feedback)}"
)

if total >= LIMITE_RODADAS:
    st.warning("Você atingiu o limite de 100 rodadas.")

if st.button("🔄 Resetar métricas"):
    resetar_metricas()
    st.rerun()

st.divider()


# ---------------------------------------------------------
# ENTRADA DO USUÁRIO
# ---------------------------------------------------------
alvo_usuario = st.text_input(
    "O que deseja que a IA reconheça?",
    placeholder="Ex.: gato, porco, abacaxi, violão...",
)

uploaded_file = st.file_uploader(
    "Escolha uma imagem",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Imagem enviada", use_container_width=True)

    except Exception:
        st.error("Não foi possível abrir a imagem.")
        st.stop()

    if not alvo_usuario.strip():
        st.warning("Digite o que deseja reconhecer.")
        st.stop()

    if total >= LIMITE_RODADAS:
        st.warning("Resete as métricas para iniciar outras rodadas.")
        st.stop()

    # -----------------------------------------------------
    # 1. TRADUZIR ANTES DA ANÁLISE
    # -----------------------------------------------------
    try:
        with st.spinner("Traduzindo o termo para inglês..."):
            termo_ingles = traduzir_para_ingles(alvo_usuario)

    except (ValueError, RuntimeError) as erro:
        st.error(str(erro))
        st.stop()

    alvo_normalizado = normalizar(termo_ingles)

    st.info(
        f"Termo original: **{alvo_usuario}**\n\n"
        f"Tradução para inglês: **{termo_ingles}**"
    )

    # -----------------------------------------------------
    # 2. ANALISAR A IMAGEM COM A IA
    # -----------------------------------------------------
    with st.spinner("Analisando a imagem..."):
        img_resized = img.resize((224, 224))

        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]

    # -----------------------------------------------------
    # 3. APLICAR O HISTÓRICO DE FEEDBACK
    # -----------------------------------------------------
    chave_termo = alvo_normalizado

    memoria = feedback.get(
        chave_termo,
        {"positivos": {}, "negativos": {}},
    )

    positivos = memoria.get("positivos", {})
    negativos = memoria.get("negativos", {})

    previsoes = []

    for _, label, prob in decoded:
        label_en = label.replace("_", " ").lower()
        label_normalizado = normalizar(label_en)

        # Ajuste heurístico com base no feedback anterior.
        # Não altera os pesos da rede neural.
        pontuacao = float(prob)

        pontuacao += 0.20 * positivos.get(label_normalizado, 0)
        pontuacao -= 0.20 * negativos.get(label_normalizado, 0)

        pontuacao = max(0.0, min(1.0, pontuacao))

        previsoes.append({
            "label_en": label_en,
            "label_normalizado": label_normalizado,
            "prob_original": float(prob),
            "pontuacao": pontuacao,
        })

    previsoes.sort(
        key=lambda item: item["pontuacao"],
        reverse=True,
    )

    # -----------------------------------------------------
    # 4. COMPARAR O TERMO COM AS PREVISÕES
    # -----------------------------------------------------
    encontrou = False

    for item in previsoes:
        label = item["label_normalizado"]

        # Correspondência textual simples.
        if (
            alvo_normalizado == label
            or alvo_normalizado in label
            or label in alvo_normalizado
        ):
            encontrou = True
            break

    # -----------------------------------------------------
    # 5. MOSTRAR RESULTADO
    # -----------------------------------------------------
    if encontrou:
        st.success(
            f"A IA encontrou uma correspondência para "
            f"**{alvo_usuario}**!"
        )
    else:
        st.error(
            f"A IA não encontrou uma correspondência textual "
            f"para **{alvo_usuario}** entre as previsões."
        )

    st.caption(
        "A correspondência é baseada nos rótulos do modelo, "
        "não em uma compreensão universal do conteúdo da imagem."
    )

    with st.expander("🔎 Ver as cinco previsões da IA"):
        for i, item in enumerate(previsoes, start=1):
            label_pt = traduzir_para_portugues(item["label_en"])

            st.write(
                f"**{i}. {item['label_en']}** "
                f"(*{label_pt}*) — "
                f"Confiança original: "
                f"{item['prob_original'] * 100:.2f}%"
            )

            if item["pontuacao"] != item["prob_original"]:
                st.caption(
                    "Pontuação ajustada pelo histórico: "
                    f"{item['pontuacao'] * 100:.2f}%"
                )

    st.divider()

    # -----------------------------------------------------
    # 6. FEEDBACK DO USUÁRIO
    # -----------------------------------------------------
    st.write("**A IA identificou corretamente o que você pediu?**")

    col_sim, col_nao = st.columns(2)

    # A chave da avaliação evita registrar a mesma avaliação
    # novamente se o usuário provocar um rerun.
    chave_avaliacao = (
        f"{uploaded_file.name}|"
        f"{uploaded_file.size}|"
        f"{normalizar(alvo_usuario)}"
    )

    with col_sim:
        if st.button("✅ Sim, acertou", use_container_width=True):
            if st.session_state.ultima_avaliacao != chave_avaliacao:
                memoria = feedback.setdefault(
                    chave_termo,
                    {"positivos": {}, "negativos": {}},
                )

                # Reforça rótulos textualmente compatíveis com o alvo.
                rotulos_compativeis = [
                    item["label_normalizado"]
                    for item in previsoes
                    if (
                        alvo_normalizado == item["label_normalizado"]
                        or alvo_normalizado in item["label_normalizado"]
                        or item["label_normalizado"] in alvo_normalizado
                    )
                ]

                if rotulos_compativeis:
                    for label in rotulos_compativeis:
                        memoria["positivos"][label] = (
                            memoria["positivos"].get(label, 0) + 1
                        )
                        memoria["negativos"].pop(label, None)
                else:
                    # Sem um rótulo compatível, não sabemos qual
                    # previsão o usuário considerou correta.
                    st.warning(
                        "Não foi possível associar esse acerto a um "
                        "rótulo específico. O resultado será contado "
                        "nas métricas, mas não ensinará um novo rótulo."
                    )

                salvar_feedback(feedback)
                registrar_resultado(correto=True)
                st.session_state.ultima_avaliacao = chave_avaliacao

                st.success("Avaliação registrada.")
                st.rerun()
            else:
                st.info("Esta imagem já foi avaliada nesta sessão.")

    with col_nao:
        if st.button("❌ Não, errou", use_container_width=True):
            if st.session_state.ultima_avaliacao != chave_avaliacao:
                memoria = feedback.setdefault(
                    chave_termo,
                    {"positivos": {}, "negativos": {}},
                )

                # Registra os rótulos exibidos como não úteis
                # para este termo específico.
                for item in previsoes:
                    label = item["label_normalizado"]

                    memoria["negativos"][label] = (
                        memoria["negativos"].get(label, 0) + 1
                    )
                    memoria["positivos"].pop(label, None)

                salvar_feedback(feedback)
                registrar_resultado(correto=False)
                st.session_state.ultima_avaliacao = chave_avaliacao

                st.success(
                    "Erro registrado. O histórico será considerado "
                    "nas próximas análises deste termo."
                )
                st.rerun()
            else:
                st.info("Esta imagem já foi avaliada nesta sessão.")
