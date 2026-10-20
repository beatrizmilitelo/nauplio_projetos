
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from deep_translator import GoogleTranslator, MyMemoryTranslator
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)


# =========================================================
# CONFIGURAÇÕES
# =========================================================

st.set_page_config(
    page_title="Detector Universal de Imagens com IA",
    page_icon="🤖",
    layout="centered",
)

PASTA = Path(__file__).resolve().parent

ARQUIVO_METRICAS = PASTA / "metricas_ia.json"
ARQUIVO_FEEDBACK = PASTA / "feedback_ia.json"
ARQUIVO_AVALIACOES = PASTA / "avaliacoes_registradas.json"

METRICAS_INICIAIS = {
    "total": 0,
    "acertos": 0,
    "erros": 0,
}

IDIOMAS = {
    "Detectar automaticamente": "auto",
    "Português (Brasil)": "pt-BR",
    "Inglês": "en",
    "Espanhol": "es",
    "Francês": "fr",
    "Alemão": "de",
    "Italiano": "it",
    "Japonês": "ja",
    "Chinês simplificado": "zh-CN",
    "Russo": "ru",
    "Árabe": "ar",
    "Hindi": "hi",
}


# =========================================================
# LEITURA E GRAVAÇÃO DE JSON
# =========================================================

def ler_json(caminho, valor_padrao):
    try:
        if caminho.exists():
            with open(caminho, "r", encoding="utf-8") as arquivo:
                return json.load(arquivo)
    except (OSError, json.JSONDecodeError):
        pass

    return valor_padrao.copy() if isinstance(valor_padrao, dict) else valor_padrao


def salvar_json(caminho, dados):
    """
    Salva o arquivo de forma segura para reduzir o risco
    de corromper os dados caso a gravação seja interrompida.
    """
    temporario = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=PASTA,
            delete=False,
            suffix=".tmp",
        ) as arquivo:
            json.dump(
                dados,
                arquivo,
                ensure_ascii=False,
                indent=2,
            )
            temporario = arquivo.name

        os.replace(temporario, caminho)
        return True

    except OSError:
        if temporario and os.path.exists(temporario):
            try:
                os.remove(temporario)
            except OSError:
                pass

        return False


def carregar_metricas():
    dados = ler_json(ARQUIVO_METRICAS, METRICAS_INICIAIS)

    try:
        return {
            "total": int(dados.get("total", 0)),
            "acertos": int(dados.get("acertos", 0)),
            "erros": int(dados.get("erros", 0)),
        }
    except (ValueError, TypeError):
        return METRICAS_INICIAIS.copy()


# =========================================================
# TRADUÇÃO
# =========================================================

def normalizar(texto):
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )
    texto = re.sub(r"[_\-]+", " ", texto)
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    return " ".join(texto.split())


@st.cache_data(ttl=86400, show_spinner=False)
def traduzir_para_ingles(texto, idioma_origem):
    """
    Tenta traduzir usando Google e, se falhar,
    tenta MyMemory.

    As traduções bem-sucedidas ficam em cache por 24 horas.
    """

    texto = texto.strip()

    if not texto:
        raise ValueError("Digite o nome de um animal ou objeto.")

    if idioma_origem == "auto":
        origem_google = "auto"
        origem_mymemory = "pt-BR"
    else:
        origem_google = idioma_origem
        origem_mymemory = idioma_origem

    erros = []

    # Primeira tentativa: Google
    try:
        resultado = GoogleTranslator(
            source=origem_google,
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip()

    except Exception as erro:
        erros.append(f"Google: {str(erro)[:250]}")

    # Segunda tentativa: MyMemory
    try:
        resultado = MyMemoryTranslator(
            source=origem_mymemory,
            target="en-GB",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip()

    except Exception as erro:
        erros.append(f"MyMemory: {str(erro)[:250]}")

    raise RuntimeError(
        "Não foi possível traduzir o texto neste momento. "
        "O serviço pode estar temporariamente indisponível. "
        "Aguarde alguns minutos e tente novamente. "
        + " | ".join(erros)
    )


# =========================================================
# MODELO DE INTELIGÊNCIA ARTIFICIAL
# =========================================================

@st.cache_resource(show_spinner="Carregando modelo de inteligência artificial...")
def carregar_modelo():
    return MobileNetV2(weights="imagenet")


# =========================================================
# ANÁLISE DE IMAGEM
# =========================================================

def preparar_imagem(arquivo):
    imagem = Image.open(arquivo).convert("RGB")
    imagem_redimensionada = imagem.resize((224, 224))

    array = np.asarray(imagem_redimensionada, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)

    return imagem, array


def pontuar_previsoes(previsoes, termo, feedback):
    """
    Ajusta heuristicamente as pontuações com base no feedback anterior.
    Isso não altera nem retreina os pesos do MobileNetV2.
    """

    termo_normalizado = normalizar(termo)
    dados_termo = feedback.get(termo_normalizado, {})

    positivos = dados_termo.get("positivos", {})
    negativos = dados_termo.get("negativos", {})

    resultado = []

    for rotulo, confianca in previsoes:
        rotulo_normalizado = normalizar(rotulo)

        pontos_positivos = int(positivos.get(rotulo_normalizado, 0))
        pontos_negativos = int(negativos.get(rotulo_normalizado, 0))

        # Ajuste limitado para que o feedback não domine a previsão.
        ajuste = min(pontos_positivos, 5) * 0.02
        ajuste -= min(pontos_negativos, 5) * 0.02

        nova_confianca = max(
            0.0,
            min(1.0, confianca + ajuste),
        )

        resultado.append(
            {
                "rotulo": rotulo,
                "confianca": confianca,
                "pontuacao_ajustada": nova_confianca,
            }
        )

    resultado.sort(
        key=lambda item: item["pontuacao_ajustada"],
        reverse=True,
    )

    return resultado


def verificar_correspondencia(termo, rotulo):
    """
    Comparação textual simples entre o termo traduzido
    e o rótulo produzido pelo modelo.
    """

    termo_normalizado = normalizar(termo)
    rotulo_normalizado = normalizar(rotulo)

    if not termo_normalizado or not rotulo_normalizado:
        return False

    palavras_termo = set(termo_normalizado.split())
    palavras_rotulo = set(rotulo_normalizado.split())

    if termo_normalizado == rotulo_normalizado:
        return True

    # Evita aceitar palavras muito curtas como correspondência.
    if len(termo_normalizado) >= 4:
        if termo_normalizado in rotulo_normalizado:
            return True

    return bool(palavras_termo.intersection(palavras_rotulo))


def criar_chave_avaliacao(texto_original, idioma, bytes_imagem):
    """
    A mesma combinação de texto, idioma e imagem gera a mesma chave.
    Isso ajuda a impedir que uma avaliação já registrada conte novamente.
    """

    hash_imagem = hashlib.sha256(bytes_imagem).hexdigest()

    identificador = (
        normalizar(texto_original)
        + "|"
        + idioma
        + "|"
        + hash_imagem
    )

    return hashlib.sha256(
        identificador.encode("utf-8")
    ).hexdigest()


# =========================================================
# REGISTRO DO FEEDBACK E MÉTRICAS
# =========================================================

def registrar_feedback(chave, termo, previsoes, acertou):
    metricas = carregar_metricas()
    avaliacoes = ler_json(ARQUIVO_AVALIACOES, {})
    feedback = ler_json(ARQUIVO_FEEDBACK, {})

    # Não contabiliza novamente uma avaliação já respondida.
    if chave in avaliacoes:
        return False, "Esta avaliação já foi registrada anteriormente."

    termo_normalizado = normalizar(termo)

    if termo_normalizado not in feedback:
        feedback[termo_normalizado] = {
            "positivos": {},
            "negativos": {},
        }

    dados_termo = feedback[termo_normalizado]

    if acertou:
        metricas["acertos"] += 1

        # Reforça o rótulo que apareceu primeiro na previsão.
        rotulo = normalizar(previsoes[0]["rotulo"])
        positivos = dados_termo["positivos"]
        positivos[rotulo] = int(positivos.get(rotulo, 0)) + 1

    else:
        metricas["erros"] += 1

        # Registra os rótulos apresentados como incorretos.
        negativos = dados_termo["negativos"]

        for previsao in previsoes:
            rotulo = normalizar(previsao["rotulo"])
            negativos[rotulo] = int(negativos.get(rotulo, 0)) + 1

    metricas["total"] = metricas["acertos"] + metricas["erros"]

    avaliacoes[chave] = {
        "termo": termo,
        "resultado": "acerto" if acertou else "erro",
    }

    metricas_salvas = salvar_json(ARQUIVO_METRICAS, metricas)
    feedback_salvo = salvar_json(ARQUIVO_FEEDBACK, feedback)
    avaliacoes_salvas = salvar_json(ARQUIVO_AVALIACOES, avaliacoes)

    if not (
        metricas_salvas
        and feedback_salvo
        and avaliacoes_salvas
    ):
        return (
            False,
            "Não foi possível salvar todos os arquivos. "
            "Verifique as permissões de gravação do ambiente.",
        )

    return True, "Feedback registrado com sucesso!"


# =========================================================
# INTERFACE
# =========================================================

st.title("Detector Universal de Imagens com IA 🤖")

st.write(
    "Digite o nome de um animal ou objeto, envie uma imagem "
    "e veja as previsões do modelo."
)

st.info(
    "O modelo MobileNetV2 foi treinado com categorias do ImageNet. "
    "Ele não consegue reconhecer todos os objetos existentes."
)

texto_digitado = st.text_input(
    "Qual animal ou objeto você quer identificar?",
    placeholder="Ex.: tartaruga, cachorro, carro, cadeira...",
)

idioma_selecionado = st.selectbox(
    "Idioma do texto digitado",
    options=list(IDIOMAS.keys()),
    index=1,
    help=(
        "Escolha o idioma do texto para ajudar a tradução. "
        "A detecção automática também está disponível."
    ),
)

arquivo_imagem = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"],
)

if arquivo_imagem is not None:
    try:
        imagem_visualizacao = Image.open(arquivo_imagem).convert("RGB")
        st.image(
            imagem_visualizacao,
            caption="Imagem enviada",
            use_container_width=True,
        )
    except Exception:
        st.error("Não foi possível abrir essa imagem.")

if st.button("🔍 Analisar imagem", type="primary", use_container_width=True):
    if not texto_digitado.strip():
        st.warning("Digite o nome de um animal ou objeto.")

    elif arquivo_imagem is None:
        st.warning("Envie uma imagem antes de iniciar a análise.")

    else:
        try:
            bytes_imagem = arquivo_imagem.getvalue()
            idioma_codigo = IDIOMAS[idioma_selecionado]

            with st.spinner("Traduzindo o texto..."):
                texto_em_ingles = traduzir_para_ingles(
                    texto_digitado,
                    idioma_codigo,
                )

            st.session_state["traducao_atual"] = texto_em_ingles

            with st.spinner("Analisando a imagem com a IA..."):
                imagem, imagem_processada = preparar_imagem(
                    arquivo_imagem
                )

                modelo = carregar_modelo()
                resultado_modelo = modelo.predict(
                    imagem_processada,
                    verbose=0,
                )

                previsoes_brutas = decode_predictions(
                    resultado_modelo,
                    top=5,
                )[0]

                previsoes = [
                    (rotulo.replace("_", " "), float(probabilidade))
                    for _, rotulo, probabilidade in previsoes_brutas
                ]

                feedback_atual = ler_json(ARQUIVO_FEEDBACK, {})

                previsoes_ajustadas = pontuar_previsoes(
                    previsoes,
                    texto_em_ingles,
                    feedback_atual,
                )

                corresponde = any(
                    verificar_correspondencia(
                        texto_em_ingles,
                        item["rotulo"],
                    )
                    for item in previsoes_ajustadas
                )

                chave = criar_chave_avaliacao(
                    texto_digitado,
                    idioma_codigo,
                    bytes_imagem,
                )

                avaliacoes_anteriores = ler_json(
                    ARQUIVO_AVALIACOES,
                    {},
                )

                st.session_state["analise_atual"] = {
                    "texto_original": texto_digitado,
                    "texto_ingles": texto_em_ingles,
                    "previsoes": previsoes_ajustadas,
                    "corresponde": corresponde,
                    "chave": chave,
                    "ja_registrada": chave in avaliacoes_anteriores,
                }

        except ValueError as erro:
            st.error(str(erro))

        except RuntimeError as erro:
            st.error(str(erro))

        except Exception as erro:
            st.error(
                "Ocorreu um erro durante a análise. "
                "Verifique a imagem e tente novamente."
            )
            st.caption(f"Detalhes técnicos: {str(erro)[:500]}")


# =========================================================
# RESULTADO DA ANÁLISE
# =========================================================

if "analise_atual" in st.session_state:
    analise = st.session_state["analise_atual"]

    st.divider()
    st.subheader("Resultado da análise")

    st.write("**Texto original:**", analise["texto_original"])
    st.write("**Texto traduzido para inglês:**", analise["texto_ingles"])

    st.write("**As principais previsões foram:**")

    for indice, previsao in enumerate(analise["previsoes"], start=1):
        confianca = previsao["confianca"] * 100

        st.write(
            f"{indice}. **{previsao['rotulo'].title()}** "
            f"— confiança original: {confianca:.2f}%"
        )

    if analise["corresponde"]:
        st.success(
            "Foi encontrada uma correspondência textual entre "
            "o termo procurado e pelo menos um dos rótulos previstos."
        )
    else:
        st.warning(
            "O termo procurado não corresponde textualmente "
            "às cinco principais previsões."
        )

    st.caption(
        "A correspondência é uma comparação de palavras, não uma "
        "confirmação semântica. A confiança é a pontuação original "
        "do modelo; o feedback não modifica os pesos do modelo."
    )

    st.divider()
    st.subheader("A IA acertou?")

    if analise["ja_registrada"]:
        st.info(
            "Esta combinação de imagem e texto já foi avaliada. "
            "Ela não será contabilizada novamente."
        )

    else:
        st.write(
            "Seu feedback será salvo para atualizar as métricas "
            "e ajustar heuristicamente as pontuações em análises futuras."
        )

        coluna_sim, coluna_nao = st.columns(2)

        with coluna_sim:
            if st.button(
                "✅ Sim, acertou",
                use_container_width=True,
            ):
                sucesso, mensagem = registrar_feedback(
                    analise["chave"],
                    analise["texto_ingles"],
                    analise["previsoes"],
                    True,
                )

                if sucesso:
                    st.success(mensagem)
                    st.session_state["analise_atual"]["ja_registrada"] = True
                    st.rerun()
                else:
                    st.error(mensagem)

        with coluna_nao:
            if st.button(
                "❌ Não, errou",
                use_container_width=True,
            ):
                sucesso, mensagem = registrar_feedback(
                    analise["chave"],
                    analise["texto_ingles"],
                    analise["previsoes"],
                    False,
                )

                if sucesso:
                    st.success(mensagem)
                    st.session_state["analise_atual"]["ja_registrada"] = True
                    st.rerun()
                else:
                    st.error(mensagem)


# =========================================================
# MÉTRICAS ACUMULADAS
# =========================================================

st.divider()
st.header("📊 Métricas acumuladas da IA")

metricas = carregar_metricas()

total = metricas["total"]
acertos = metricas["acertos"]
erros = metricas["erros"]

if total > 0:
    percentual_acertos = (acertos / total) * 100
    percentual_erros = (erros / total) * 100
else:
    percentual_acertos = 0.0
    percentual_erros = 0.0

coluna_total, coluna_acertos, coluna_erros = st.columns(3)

coluna_total.metric("Avaliações", total)
coluna_acertos.metric("Acertos", acertos)
coluna_erros.metric("Erros", erros)

coluna_pct_acertos, coluna_pct_erros = st.columns(2)

coluna_pct_acertos.metric(
    "Percentual de acertos",
    f"{percentual_acertos:.2f}%",
)

coluna_pct_erros.metric(
    "Percentual de erros",
    f"{percentual_erros:.2f}%",
)

st.caption(
    "As métricas consideram apenas avaliações em que você clicou "
    "em 'Sim, acertou' ou 'Não, errou'."
)

with st.expander("Como essas métricas funcionam?"):
    st.write(
        "- Cada avaliação confirmada conta uma vez.\n"
        "- Acertos e erros são registrados em arquivos JSON.\n"
        "- O cache de tradução reduz chamadas repetidas aos serviços externos.\n"
        "- O feedback salvo ajusta pontuações de forma heurística.\n"
        "- O modelo MobileNetV2 não é retreinado automaticamente."
    )

if st.button("🗑️ Zerar métricas e histórico de avaliações"):
    metricas_zeradas = METRICAS_INICIAIS.copy()

    metricas_ok = salvar_json(
        ARQUIVO_METRICAS,
        metricas_zeradas,
    )

    avaliacoes_ok = salvar_json(
        ARQUIVO_AVALIACOES,
        {},
    )

    if metricas_ok and avaliacoes_ok:
        st.success(
            "Métricas e histórico de avaliações zerados. "
            "O feedback heurístico foi mantido."
        )
        st.rerun()
    else:
        st.error(
            "Não foi possível zerar os arquivos. "
            "Verifique as permissões de gravação."
        )
