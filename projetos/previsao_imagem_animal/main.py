
import os
import re
import json
import hashlib
import unicodedata

import streamlit as st
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator, MyMemoryTranslator
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image


# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================

st.set_page_config(
    page_title="Detector Universal por IA",
    page_icon="🤖",
    layout="centered",
)

ARQUIVO_METRICAS = "metricas_ia.json"
ARQUIVO_FEEDBACK = "feedback_ia.json"

st.title("Detector Universal de Imagens com IA 🤖")

st.write(
    "Digite o nome de um animal ou objeto, envie uma imagem "
    "e avalie o resultado da inteligência artificial."
)


# =========================================================
# 2. FUNÇÕES DE PERSISTÊNCIA
# =========================================================

def salvar_json(caminho, dados):
    """Salva os dados em JSON usando arquivo temporário."""
    temporario = caminho + ".tmp"

    with open(temporario, "w", encoding="utf-8") as arquivo:
        json.dump(
            dados,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temporario, caminho)


def carregar_metricas():
    """Recupera as métricas acumuladas entre sessões."""
    padrao = {
        "total_rodadas": 0,
        "acertos": 0,
        "erros": 0,
    }

    if not os.path.exists(ARQUIVO_METRICAS):
        return padrao

    try:
        with open(ARQUIVO_METRICAS, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        metricas = {
            "total_rodadas": max(0, int(dados.get("total_rodadas", 0))),
            "acertos": max(0, int(dados.get("acertos", 0))),
            "erros": max(0, int(dados.get("erros", 0))),
        }

        # Garante consistência com os totais salvos.
        if metricas["total_rodadas"] != (
            metricas["acertos"] + metricas["erros"]
        ):
            metricas["total_rodadas"] = (
                metricas["acertos"] + metricas["erros"]
            )

        return metricas

    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        st.warning(
            "Não foi possível ler o arquivo das métricas. "
            "Verifique o arquivo metricas_ia.json."
        )
        return padrao


def salvar_metricas():
    """Persiste os contadores após cada avaliação."""
    dados = {
        "total_rodadas": st.session_state.total_rodadas,
        "acertos": st.session_state.acertos,
        "erros": st.session_state.erros,
    }

    salvar_json(ARQUIVO_METRICAS, dados)


def carregar_feedback():
    """Carrega o histórico de aprendizado por termo."""
    if not os.path.exists(ARQUIVO_FEEDBACK):
        return {}

    try:
        with open(ARQUIVO_FEEDBACK, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        return dados if isinstance(dados, dict) else {}

    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        st.warning(
            "Não foi possível ler o histórico de feedback."
        )
        return {}


# =========================================================
# 3. INICIALIZAÇÃO DA SESSÃO
# =========================================================

if "metricas_carregadas" not in st.session_state:
    metricas = carregar_metricas()

    st.session_state.total_rodadas = metricas["total_rodadas"]
    st.session_state.acertos = metricas["acertos"]
    st.session_state.erros = metricas["erros"]

    st.session_state.metricas_carregadas = True

if "ultima_avaliacao" not in st.session_state:
    st.session_state.ultima_avaliacao = None

if "resultado_atual" not in st.session_state:
    st.session_state.resultado_atual = None


# =========================================================
# 4. NORMALIZAÇÃO DE TEXTO
# =========================================================

def normalizar(texto):
    texto = str(texto).lower().strip()

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    texto = texto.replace("_", " ")
    texto = re.sub(r"[^\w\s-]", " ", texto)

    return re.sub(r"\s+", " ", texto).strip()


# =========================================================
# 5. TRADUÇÃO PARA INGLÊS
# =========================================================

def traduzir_para_ingles(texto):
    texto = texto.strip()

    if not texto:
        raise ValueError(
            "Digite o nome do animal ou objeto."
        )

    erros = []

    # Primeira tentativa: detectar qualquer idioma.
    try:
        resultado = GoogleTranslator(
            source="auto",
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip().lower()

    except Exception as erro:
        erros.append(f"Google automático: {erro}")

    # Segunda tentativa: português para inglês.
    try:
        resultado = GoogleTranslator(
            source="pt",
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip().lower()

    except Exception as erro:
        erros.append(f"Google PT-EN: {erro}")

    # Terceira tentativa: serviço alternativo.
    # Essa alternativa considera o texto em português.
    try:
        resultado = MyMemoryTranslator(
            source="pt",
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip().lower()

    except Exception as erro:
        erros.append(f"MyMemory PT-EN: {erro}")

    detalhes = "\n".join(erros)

    raise RuntimeError(
        "Não foi possível traduzir o texto.\n\n"
        "Verifique se o ambiente tem acesso à internet. "
        "Se estiver hospedado, o servidor pode estar bloqueando "
        "as requisições aos serviços de tradução.\n\n"
        f"Detalhes técnicos:\n{detalhes}"
    )


def traduzir_para_portugues(texto):
    """Tradução opcional para exibir as previsões."""
    try:
        resultado = GoogleTranslator(
            source="en",
            target="pt",
        ).translate(texto)

        return resultado.lower().strip() if resultado else texto

    except Exception:
        return texto


# =========================================================
# 6. CARREGAR MODELO DE IA
# =========================================================

@st.cache_resource
def carregar_modelo():
    return MobileNetV2(weights="imagenet")


with st.spinner("Carregando o modelo de inteligência artificial..."):
    model = carregar_modelo()


# =========================================================
# 7. REGISTRAR AVALIAÇÕES
# =========================================================

def registrar_resultado(correto):
    """Cada avaliação confirmada é contabilizada e salva."""
    st.session_state.total_rodadas += 1

    if correto:
        st.session_state.acertos += 1
    else:
        st.session_state.erros += 1

    salvar_metricas()


# =========================================================
# 8. ENTRADA DA IMAGEM
# =========================================================

alvo_usuario = st.text_input(
    "O que você quer que a IA reconheça?",
    placeholder="Ex.: gato, tartaruga, baleia, abacaxi...",
)

uploaded_file = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"],
)

feedback = carregar_feedback()


# =========================================================
# 9. PROCESSAMENTO DA IMAGEM
# =========================================================

if uploaded_file is not None:

    try:
        img = Image.open(uploaded_file).convert("RGB")

        st.image(
            img,
            caption="Imagem enviada",
            use_container_width=True,
        )

    except Exception:
        st.error("Não foi possível abrir essa imagem.")
        img = None

    if img is not None:

        if not alvo_usuario.strip():
            st.info(
                "Digite o nome do animal ou objeto que deseja reconhecer."
            )

        else:

            # ---------------------------------------------
            # 9.1 TRADUZIR ANTES DE ANALISAR
            # ---------------------------------------------

            try:
                with st.spinner("Traduzindo para inglês..."):
                    termo_ingles = traduzir_para_ingles(alvo_usuario)

            except (ValueError, RuntimeError) as erro:
                st.error(str(erro))
                termo_ingles = None

            # Prossegue apenas se a tradução funcionar.
            if termo_ingles:

                st.write(
                    f"**Termo original:** {alvo_usuario}"
                )

                st.write(
                    f"**Tradução para inglês:** {termo_ingles}"
                )

                # -----------------------------------------
                # 9.2 PREPARAR E ANALISAR A IMAGEM
                # -----------------------------------------

                try:
                    with st.spinner("Analisando a imagem..."):
                        img_resized = img.resize((224, 224))

                        x = keras_image.img_to_array(img_resized)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)

                        preds = model.predict(x, verbose=0)

                        decoded = decode_predictions(
                            preds,
                            top=5,
                        )[0]

                except Exception as erro:
                    st.error(
                        f"Não foi possível analisar a imagem: {erro}"
                    )
                    decoded = None

                if decoded is not None:

                    # -------------------------------------
                    # 9.3 APLICAR FEEDBACK ANTERIOR
                    # -------------------------------------

                    termo = normalizar(termo_ingles)

                    memoria = feedback.get(
                        termo,
                        {
                            "positivos": {},
                            "negativos": {},
                        },
                    )

                    positivos = memoria.get("positivos", {})
                    negativos = memoria.get("negativos", {})

                    previsoes = []

                    for _, label, prob in decoded:
                        label_en = label.replace("_", " ").lower()
                        label_norm = normalizar(label_en)

                        pontuacao = float(prob)

                        # Ajuste heurístico usando feedback salvo.
                        pontuacao += (
                            0.20 * positivos.get(label_norm, 0)
                        )

                        pontuacao -= (
                            0.20 * negativos.get(label_norm, 0)
                        )

                        pontuacao = max(
                            0.0,
                            min(1.0, pontuacao),
                        )

                        previsoes.append({
                            "label": label_en,
                            "normalizado": label_norm,
                            "probabilidade": float(prob),
                            "pontuacao": pontuacao,
                        })

                    previsoes.sort(
                        key=lambda item: item["pontuacao"],
                        reverse=True,
                    )

                    # -------------------------------------
                    # 9.4 VERIFICAR CORRESPONDÊNCIA
                    # -------------------------------------

                    encontrou = any(
                        termo == item["normalizado"]
                        or termo in item["normalizado"]
                        or item["normalizado"] in termo
                        for item in previsoes
                    )

                    if encontrou:
                        st.success(
                            f"Foi encontrada uma correspondência "
                            f"para **{alvo_usuario}**!"
                        )
                    else:
                        st.error(
                            f"Não foi encontrada correspondência textual "
                            f"para **{alvo_usuario}** nas cinco previsões."
                        )

                    st.caption(
                        "A correspondência é textual. O modelo original "
                        "reconhece categorias do ImageNet, não todos os "
                        "animais e objetos existentes."
                    )

                    # -------------------------------------
                    # 9.5 EXIBIR PREVISÕES
                    # -------------------------------------

                    with st.expander("🔎 Ver as cinco previsões da IA"):
                        for i, item in enumerate(previsoes, 1):
                            label_pt = traduzir_para_portugues(
                                item["label"]
                            )

                            st.write(
                                f"**{i}. {item['label']}** "
                                f"(*{label_pt}*) — "
                                f"Confiança original: "
                                f"{item['probabilidade'] * 100:.2f}%"
                            )

                            if (
                                item["pontuacao"]
                                != item["probabilidade"]
                            ):
                                st.caption(
                                    "Pontuação ajustada pelo histórico: "
                                    f"{item['pontuacao'] * 100:.2f}%"
                                )

                    st.divider()

                    # -------------------------------------
                    # 9.6 IDENTIFICAR A AVALIAÇÃO
                    # -------------------------------------

                    hash_imagem = hashlib.sha256(
                        uploaded_file.getvalue()
                    ).hexdigest()

                    chave_avaliacao = (
                        f"{hash_imagem}|{termo}"
                    )

                    st.write("**A IA acertou?**")

                    col_sim, col_nao = st.columns(2)

                    # -------------------------------------
                    # 9.7 BOTÃO DE ACERTO
                    # -------------------------------------

                    with col_sim:
                        if st.button(
                            "✅ Sim, acertou",
                            use_container_width=True,
                        ):

                            if (
                                st.session_state.ultima_avaliacao
                                == chave_avaliacao
                            ):
                                st.warning(
                                    "Essa avaliação já foi registrada "
                                    "nesta sessão."
                                )

                            else:
                                memoria = feedback.setdefault(
                                    termo,
                                    {
                                        "positivos": {},
                                        "negativos": {},
                                    },
                                )

                                compativeis = [
                                    item["normalizado"]
                                    for item in previsoes
                                    if (
                                        termo == item["normalizado"]
                                        or termo in item["normalizado"]
                                        or item["normalizado"] in termo
                                    )
                                ]

                                if compativeis:
                                    for label in compativeis:
                                        memoria["positivos"][label] = (
                                            memoria["positivos"].get(
                                                label, 0
                                            ) + 1
                                        )

                                        memoria["negativos"].pop(
                                            label, None
                                        )

                                    salvar_json(
                                        ARQUIVO_FEEDBACK,
                                        feedback,
                                    )

                                registrar_resultado(True)

                                st.session_state.ultima_avaliacao = (
                                    chave_avaliacao
                                )

                                st.success("Acerto registrado!")
                                st.rerun()

                    # -------------------------------------
                    # 9.8 BOTÃO DE ERRO
                    # -------------------------------------

                    with col_nao:
                        if st.button(
                            "❌ Não, errou",
                            use_container_width=True,
                        ):

                            if (
                                st.session_state.ultima_avaliacao
                                == chave_avaliacao
                            ):
                                st.warning(
                                    "Essa avaliação já foi registrada "
                                    "nesta sessão."
                                )

                            else:
                                memoria = feedback.setdefault(
                                    termo,
                                    {
                                        "positivos": {},
                                        "negativos": {},
                                    },
                                )

                                for item in previsoes:
                                    label = item["normalizado"]

                                    memoria["negativos"][label] = (
                                        memoria["negativos"].get(
                                            label, 0
                                        ) + 1
                                    )

                                    memoria["positivos"].pop(
                                        label, None
                                    )

                                salvar_json(
                                    ARQUIVO_FEEDBACK,
                                    feedback,
                                )

                                registrar_resultado(False)

                                st.session_state.ultima_avaliacao = (
                                    chave_avaliacao
                                )

                                st.success(
                                    "Erro registrado no histórico!"
                                )

                                st.rerun()


# =========================================================
# 10. MÉTRICAS ACUMULADAS NO FINAL DA PÁGINA
# =========================================================

st.divider()
st.subheader("📊 Métricas acumuladas")
st.caption(
    "Todas as avaliações registradas são contabilizadas, "
    "inclusive após fechar e reabrir o aplicativo."
)

total = st.session_state.total_rodadas
acertos = st.session_state.acertos
erros = st.session_state.erros

pct_acertos = (
    acertos / total * 100
    if total > 0
    else 0.0
)

pct_erros = (
    erros / total * 100
    if total > 0
    else 0.0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total de tentativas",
    total,
)

col2.metric(
    "Acertos",
    acertos,
    f"{pct_acertos:.2f}%",
)

col3.metric(
    "Erros",
    erros,
    f"{pct_erros:.2f}%",
    delta_color="inverse",
)

col4.metric(
    "Acurácia total",
    f"{pct_acertos:.2f}%",
)

if total > 0:
    st.progress(pct_acertos / 100)
else:
    st.progress(0.0)
    st.info(
        "As métricas serão atualizadas quando você avaliar "
        "a primeira imagem."
    )

st.caption(
    f"Conferência: {acertos} acertos + {erros} erros = "
    f"{total} tentativas."
)

# O reset é opcional e apaga apenas as métricas.
# O histórico de feedback continua preservado.
if st.button("🔄 Zerar métricas acumuladas"):

    st.session_state.total_rodadas = 0
    st.session_state.acertos = 0
    st.session_state.erros = 0
    st.session_state.ultima_avaliacao = None

    salvar_metricas()

    st.rerun()
