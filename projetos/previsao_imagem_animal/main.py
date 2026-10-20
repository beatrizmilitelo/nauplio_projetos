
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
# CONFIGURAÇÃO
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

CODIGOS_GOOGLE = {
    "pt-BR": "pt",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "ja": "ja",
    "zh-CN": "zh-CN",
    "ru": "ru",
    "ar": "ar",
    "hi": "hi",
}

CODIGOS_MYMEMORY = {
    "pt-BR": "pt-BR",
    "en": "en-GB",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "ja": "ja-JP",
    "zh-CN": "zh-CN",
    "ru": "ru-RU",
    "ar": "ar-SA",
    "hi": "hi-IN",
}


# =========================================================
# NORMALIZAÇÃO DE TEXTO
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


# =========================================================
# DICIONÁRIO LOCAL DE TRADUÇÕES
# =========================================================

TRADUCOES_CONHECIDAS = {
    # Animais
    "porco": "pig",
    "porca": "sow",
    "leitao": "piglet",
    "leitões": "piglets",
    "tartaruga": "turtle",
    "tartarugas": "turtles",
    "cachorro": "dog",
    "cachorros": "dogs",
    "cao": "dog",
    "gato": "cat",
    "gatos": "cats",
    "cavalo": "horse",
    "vaca": "cow",
    "boi": "ox",
    "galinha": "hen",
    "frango": "chicken",
    "pato": "duck",
    "rato": "mouse",
    "coelho": "rabbit",
    "macaco": "monkey",
    "elefante": "elephant",
    "leao": "lion",
    "tigre": "tiger",
    "urso": "bear",
    "cobra": "snake",
    "peixe": "fish",
    "golfinho": "dolphin",
    "baleia": "whale",
    "tubarao": "shark",
    "papagaio": "parrot",
    "borboleta": "butterfly",
    "abelha": "bee",
    "formiga": "ant",
    "aranha": "spider",
    "pinguim": "penguin",
    "coruja": "owl",
    "aguia": "eagle",
    "galinha d'angola": "guinea fowl",
    "ovelha": "sheep",
    "cabra": "goat",
    "porquinho da india": "guinea pig",
    "morcego": "bat",
    "jacare": "alligator",
    "crocodilo": "crocodile",
    "sapo": "frog",
    "lagarto": "lizard",
    "onca": "jaguar",
    "raposa": "fox",
    "lobo": "wolf",
    "veado": "deer",
    "girafa": "giraffe",
    "zebra": "zebra",
    "camelo": "camel",
    "polvo": "octopus",
    "lula": "squid",
    "cavalo marinho": "seahorse",
    "estrela do mar": "starfish",
    "tubarao branco": "great white shark",
    "tartaruga marinha": "sea turtle",

    # Objetos
    "carro": "car",
    "bicicleta": "bicycle",
    "cadeira": "chair",
    "mesa": "table",
    "celular": "mobile phone",
    "telefone": "telephone",
    "computador": "computer",
    "livro": "book",
    "copo": "glass",
    "garrafa": "bottle",
    "chave": "key",
    "relogio": "clock",
    "oculos": "glasses",
    "mochila": "backpack",
    "caneta": "pen",
    "lapis": "pencil",
    "caderno": "notebook",
    "cama": "bed",
    "sofa": "sofa",
    "janela": "window",
    "porta": "door",
    "mesa de jantar": "dining table",
    "televisao": "television",
    "controle remoto": "remote control",
    "teclado": "keyboard",
    "mouse de computador": "computer mouse",
    "fone de ouvido": "headphones",
    "aviao": "airplane",
    "barco": "boat",
    "navio": "ship",
    "moto": "motorcycle",
    "onibus": "bus",
    "caminhao": "truck",
    "trem": "train",
    "relogio de pulso": "wristwatch",
    "bola": "ball",
    "boneca": "doll",
    "brinquedo": "toy",
    "tesoura": "scissors",
    "martelo": "hammer",
    "guarda chuva": "umbrella",

    # Natureza e alimentos
    "arvore": "tree",
    "flor": "flower",
    "folha": "leaf",
    "praia": "beach",
    "mar": "sea",
    "oceano": "ocean",
    "montanha": "mountain",
    "rio": "river",
    "sol": "sun",
    "lua": "moon",
    "estrela": "star",
    "nuvem": "cloud",
    "maçã": "apple",
    "maca": "apple",
    "banana": "banana",
    "laranja": "orange",
    "morango": "strawberry",
    "uva": "grape",
    "abacaxi": "pineapple",
    "melancia": "watermelon",
    "pao": "bread",
    "queijo": "cheese",
    "ovo": "egg",
    "leite": "milk",
    "arroz": "rice",
    "feijao": "beans",
}


# =========================================================
# ARQUIVOS JSON: LEITURA E GRAVAÇÃO
# =========================================================

def ler_json(caminho, padrao):
    try:
        if caminho.exists():
            with open(caminho, "r", encoding="utf-8") as arquivo:
                return json.load(arquivo)
    except (OSError, json.JSONDecodeError):
        pass

    if isinstance(padrao, dict):
        return padrao.copy()

    return padrao


def salvar_json(caminho, dados):
    temporario = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(PASTA),
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
    dados = ler_json(
        ARQUIVO_METRICAS,
        METRICAS_INICIAIS,
    )

    try:
        acertos = int(dados.get("acertos", 0))
        erros = int(dados.get("erros", 0))

        return {
            "total": acertos + erros,
            "acertos": acertos,
            "erros": erros,
        }

    except (ValueError, TypeError, AttributeError):
        return METRICAS_INICIAIS.copy()


# =========================================================
# TRADUÇÃO
# =========================================================

@st.cache_data(ttl=86400, show_spinner=False)
def traduzir_para_ingles(texto, idioma_origem):
    texto = texto.strip()

    if not texto:
        raise ValueError(
            "Digite o nome de um animal ou objeto."
        )

    chave = normalizar(texto)

    # Evita traduções incorretas para palavras cadastradas.
    if idioma_origem in ("pt-BR", "auto"):
        if chave in TRADUCOES_CONHECIDAS:
            return TRADUCOES_CONHECIDAS[chave]

    erros = []

    # Primeira tentativa: Google Tradutor.
    try:
        origem_google = (
            "auto"
            if idioma_origem == "auto"
            else CODIGOS_GOOGLE.get(idioma_origem, idioma_origem)
        )

        resultado = GoogleTranslator(
            source=origem_google,
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip()

    except Exception as erro:
        erros.append(f"Google: {str(erro)[:180]}")

    # Segunda tentativa: MyMemory.
    # Não usamos um idioma inventado quando a origem é automática.
    if idioma_origem != "auto":
        try:
            origem_mymemory = CODIGOS_MYMEMORY.get(
                idioma_origem,
                idioma_origem,
            )

            resultado = MyMemoryTranslator(
                source=origem_mymemory,
                target="en-GB",
            ).translate(texto)

            if resultado and resultado.strip():
                return resultado.strip()

        except Exception as erro:
            erros.append(f"MyMemory: {str(erro)[:180]}")

    detalhes = " | ".join(erros)

    raise RuntimeError(
        "Não foi possível traduzir o texto. "
        "Verifique sua conexão ou tente novamente mais tarde. "
        + detalhes
    )


# =========================================================
# CARREGAR MODELO
# =========================================================

@st.cache_resource(show_spinner="Carregando o modelo de IA...")
def carregar_modelo():
    return MobileNetV2(weights="imagenet")


# =========================================================
# PROCESSAMENTO DA IMAGEM
# =========================================================

def preparar_imagem(bytes_imagem):
    imagem = Image.open(
        __import__("io").BytesIO(bytes_imagem)
    ).convert("RGB")

    imagem_redimensionada = imagem.resize((224, 224))

    array = np.asarray(
        imagem_redimensionada,
        dtype=np.float32,
    )

    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)

    return imagem, array


def prever_imagem(bytes_imagem):
    modelo = carregar_modelo()
    imagem, array = preparar_imagem(bytes_imagem)

    resultado = modelo.predict(array, verbose=0)

    previsoes_brutas = decode_predictions(
        resultado,
        top=5,
    )[0]

    previsoes = []

    for _, rotulo, probabilidade in previsoes_brutas:
        previsoes.append({
            "rotulo": rotulo.replace("_", " "),
            "confianca": float(probabilidade),
        })

    return imagem, previsoes


# =========================================================
# CORRESPONDÊNCIA DO TEXTO COM AS PREVISÕES
# =========================================================

def verificar_correspondencia(termo, rotulo):
    termo_normalizado = normalizar(termo)
    rotulo_normalizado = normalizar(rotulo)

    if not termo_normalizado or not rotulo_normalizado:
        return False

    # Correspondência exata
    if termo_normalizado == rotulo_normalizado:
        return True

    # Se o termo buscado está contido dentro do rótulo da IA (ex: "cat" dentro de "egyptian cat" ou "tiger cat")
    if termo_normalizado in rotulo_normalizado:
        return True

    palavras_termo = set(termo_normalizado.split())
    palavras_rotulo = set(rotulo_normalizado.split())

    # Se houver interseção direta de palavras maiores
    if palavras_termo.intersection(palavras_rotulo):
        return True

    return False

# =========================================================
# AJUSTE HEURÍSTICO COM FEEDBACK
# =========================================================

def pontuar_previsoes(previsoes, termo, feedback):
    termo_normalizado = normalizar(termo)

    dados = feedback.get(
        termo_normalizado,
        {"positivos": {}, "negativos": {}},
    )

    positivos = dados.get("positivos", {})
    negativos = dados.get("negativos", {})

    resultado = []

    for item in previsoes:
        rotulo = item["rotulo"]
        rotulo_normalizado = normalizar(rotulo)

        numero_positivos = int(
            positivos.get(rotulo_normalizado, 0)
        )

        numero_negativos = int(
            negativos.get(rotulo_normalizado, 0)
        )

        ajuste = min(numero_positivos, 5) * 0.02
        ajuste -= min(numero_negativos, 5) * 0.02

        pontuacao = max(
            0.0,
            min(1.0, item["confianca"] + ajuste),
        )

        resultado.append({
            "rotulo": rotulo,
            "confianca": item["confianca"],
            "pontuacao_ajustada": pontuacao,
        })

    resultado.sort(
        key=lambda item: item["pontuacao_ajustada"],
        reverse=True,
    )

    return resultado


# =========================================================
# IDENTIFICAR UMA AVALIAÇÃO
# =========================================================

def criar_chave_avaliacao(texto, idioma, bytes_imagem):
    hash_imagem = hashlib.sha256(
        bytes_imagem
    ).hexdigest()

    identificador = (
        normalizar(texto)
        + "|"
        + idioma
        + "|"
        + hash_imagem
    )

    return hashlib.sha256(
        identificador.encode("utf-8")
    ).hexdigest()


# =========================================================
# REGISTRAR FEEDBACK
# =========================================================

def registrar_feedback(chave, termo, previsoes, acertou):
    metricas = carregar_metricas()

    feedback = ler_json(
        ARQUIVO_FEEDBACK,
        {},
    )

    avaliacoes = ler_json(
        ARQUIVO_AVALIACOES,
        {},
    )

    if chave in avaliacoes:
        return (
            False,
            "Esta avaliação já foi registrada.",
        )

    termo_normalizado = normalizar(termo)

    if termo_normalizado not in feedback:
        feedback[termo_normalizado] = {
            "positivos": {},
            "negativos": {},
        }

    dados = feedback[termo_normalizado]

    if acertou:
        metricas["acertos"] += 1

        # Reforça a principal previsão apresentada.
        rotulo = normalizar(
            previsoes[0]["rotulo"]
        )

        positivos = dados["positivos"]

        positivos[rotulo] = (
            int(positivos.get(rotulo, 0)) + 1
        )

    else:
        metricas["erros"] += 1

        # Registra que as previsões apresentadas não foram aceitas.
        negativos = dados["negativos"]

        for item in previsoes:
            rotulo = normalizar(item["rotulo"])

            negativos[rotulo] = (
                int(negativos.get(rotulo, 0)) + 1
            )

    metricas["total"] = (
        metricas["acertos"] + metricas["erros"]
    )

    avaliacoes[chave] = {
        "termo": termo,
        "resultado": "acerto" if acertou else "erro",
    }

    metricas_ok = salvar_json(
        ARQUIVO_METRICAS,
        metricas,
    )

    feedback_ok = salvar_json(
        ARQUIVO_FEEDBACK,
        feedback,
    )

    avaliacoes_ok = salvar_json(
        ARQUIVO_AVALIACOES,
        avaliacoes,
    )

    if not (
        metricas_ok
        and feedback_ok
        and avaliacoes_ok
    ):
        return (
            False,
            "Não foi possível salvar todos os dados. "
            "Verifique as permissões de gravação da hospedagem.",
        )

    return True, "Feedback registrado com sucesso!"


# =========================================================
# INTERFACE PRINCIPAL
# =========================================================

st.title("Detector Universal de Imagens com IA 🤖")

st.write(
    "Digite o nome de um animal ou objeto, envie uma imagem "
    "e deixe a IA analisar."
)

st.info(
    "O MobileNetV2 utiliza categorias do ImageNet. "
    "Ele não reconhece todos os objetos do mundo."
)

texto_digitado = st.text_input(
    "Qual animal ou objeto você quer identificar?",
    placeholder="Ex.: porco, tartaruga, cachorro, carro...",
)

idioma_selecionado = st.selectbox(
    "Idioma do texto digitado",
    options=list(IDIOMAS.keys()),
    index=1,
)

arquivo_imagem = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"],
)

bytes_imagem = None

if arquivo_imagem is not None:
    bytes_imagem = arquivo_imagem.getvalue()

    try:
        imagem_preview = Image.open(
            __import__("io").BytesIO(bytes_imagem)
        ).convert("RGB")

        st.image(
            imagem_preview,
            caption="Imagem enviada",
            use_container_width=True,
        )

    except Exception:
        st.error("Não foi possível abrir a imagem enviada.")
        bytes_imagem = None


# =========================================================
# EXECUTAR A ANÁLISE
# =========================================================

if st.button(
    "🔍 Analisar imagem",
    type="primary",
    use_container_width=True,
):
    if not texto_digitado.strip():
        st.warning("Digite o nome de um animal ou objeto.")

    elif bytes_imagem is None:
        st.warning("Envie uma imagem válida antes de analisar.")

    else:
        try:
            idioma_codigo = IDIOMAS[idioma_selecionado]

            with st.spinner("Traduzindo o texto..."):
                texto_em_ingles = traduzir_para_ingles(
                    texto_digitado,
                    idioma_codigo,
                )

            with st.spinner("Analisando a imagem..."):
                imagem, previsoes_originais = prever_imagem(
                    bytes_imagem
                )

            feedback_salvo = ler_json(
                ARQUIVO_FEEDBACK,
                {},
            )

            previsoes = pontuar_previsoes(
                previsoes_originais,
                texto_em_ingles,
                feedback_salvo,
            )

            corresponde = any(
                verificar_correspondencia(
                    texto_em_ingles,
                    item["rotulo"],
                )
                for item in previsoes
            )

            chave = criar_chave_avaliacao(
                texto_digitado,
                idioma_codigo,
                bytes_imagem,
            )

            avaliacoes = ler_json(
                ARQUIVO_AVALIACOES,
                {},
            )

            st.session_state["analise_atual"] = {
                "texto_original": texto_digitado,
                "texto_ingles": texto_em_ingles,
                "previsoes": previsoes,
                "corresponde": corresponde,
                "chave": chave,
                "ja_registrada": chave in avaliacoes,
            }

        except (ValueError, RuntimeError) as erro:
            st.error(str(erro))

        except Exception as erro:
            st.error(
                "Ocorreu um erro durante a análise. "
                "Verifique a imagem, as dependências e os logs."
            )

            st.caption(
                f"Detalhes técnicos: {str(erro)[:400]}"
            )


# =========================================================
# MOSTRAR RESULTADO
# =========================================================

if "analise_atual" in st.session_state:
    analise = st.session_state["analise_atual"]

    st.divider()
    st.subheader("Resultado da análise")

    st.write(
        "**Texto original:**",
        analise["texto_original"],
    )

    st.write(
        "**Texto traduzido para inglês:**",
        analise["texto_ingles"],
    )

    st.write("### Principais previsões")

    for indice, item in enumerate(
        analise["previsoes"],
        start=1,
    ):
        st.write(
            f"{indice}. **{item['rotulo'].title()}** "
            f"— confiança original: "
            f"{item['confianca'] * 100:.2f}%"
        )

    if analise["corresponde"]:
        st.success(
            "Pelo menos uma previsão compartilha palavras "
            "com o termo procurado."
        )
    else:
        st.warning(
            "As cinco principais previsões não apresentaram "
            "uma correspondência textual com o termo procurado."
        )

    st.caption(
        "A correspondência textual não garante que o objeto foi "
        "identificado corretamente. A confiança exibida é a do "
        "modelo original, antes do ajuste heurístico."
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
            "Informe se a análise acertou. Sua resposta será "
            "registrada nas métricas."
        )

        col_sim, col_nao = st.columns(2)

        with col_sim:
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
                    st.session_state[
                        "analise_atual"
                    ]["ja_registrada"] = True

                    st.success(mensagem)
                    st.rerun()

                else:
                    st.error(mensagem)

        with col_nao:
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
                    st.session_state[
                        "analise_atual"
                    ]["ja_registrada"] = True

                    st.success(mensagem)
                    st.rerun()

                else:
                    st.error(mensagem)


# =========================================================
# MÉTRICAS ACUMULADAS
# =========================================================

st.divider()
st.header("📊 Métricas acumuladas")

metricas = carregar_metricas()

total = metricas["total"]
acertos = metricas["acertos"]
erros = metricas["erros"]

percentual_acertos = (
    acertos / total * 100
    if total > 0
    else 0.0
)

percentual_erros = (
    erros / total * 100
    if total > 0
    else 0.0
)

col_total, col_acertos, col_erros = st.columns(3)

col_total.metric(
    "Avaliações",
    total,
)

col_acertos.metric(
    "Acertos",
    acertos,
)

col_erros.metric(
    "Erros",
    erros,
)

col_pct_acertos, col_pct_erros = st.columns(2)

col_pct_acertos.metric(
    "Percentual de acertos",
    f"{percentual_acertos:.2f}%",
)

col_pct_erros.metric(
    "Percentual de erros",
    f"{percentual_erros:.2f}%",
)

st.caption(
    "As porcentagens consideram somente avaliações confirmadas "
    "pelos botões de feedback."
)

with st.expander("Como funcionam as métricas e o feedback?"):
    st.write(
        """
        - Cada combinação de imagem, texto e idioma pode ser
          contabilizada uma vez.
        - As métricas são armazenadas em arquivos JSON.
        - O feedback é utilizado para ajustar pontuações de forma
          heurística em análises futuras.
        - O feedback não treina automaticamente os pesos do
          MobileNetV2.
        - A hospedagem pode apagar arquivos locais durante
          reinicializações ou novas implantações.
        """
    )


# =========================================================
# ZERAR MÉTRICAS
# =========================================================

if st.button("🗑️ Zerar métricas e histórico de avaliações"):
    metricas_ok = salvar_json(
        ARQUIVO_METRICAS,
        METRICAS_INICIAIS.copy(),
    )

    avaliacoes_ok = salvar_json(
        ARQUIVO_AVALIACOES,
        {},
    )

    if metricas_ok and avaliacoes_ok:
        st.success(
            "Métricas e histórico de avaliações zerados. "
            "O histórico de feedback heurístico foi mantido."
        )
        st.rerun()

    else:
        st.error(
            "Não foi possível zerar os dados. "
            "Verifique as permissões de gravação."
        )
