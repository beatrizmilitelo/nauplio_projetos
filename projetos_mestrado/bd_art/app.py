import io
import os
import re
import json
import math
import zipfile
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import streamlit as st

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# CONFIGURAÇÃO DA INTERFACE
# ============================================================
st.set_page_config(
    page_title="Náuplio | Extração semântica científica",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Náuplio Analytics | Extrator científico semântico")
st.markdown(
    """
Este protótipo extrai a estrutura dos PDFs com **Docling**, usa um modelo local
via **Ollama** para identificar variáveis sem um dicionário fixo e cria
**embeddings** para comparar conceitos entre artigos.

O sistema preserva a descrição original, sugere um conceito padronizado e
registra a similaridade encontrada. A similaridade é uma sugestão, não uma
prova automática de equivalência científica.
"""
)


# ============================================================
# CONFIGURAÇÃO DO OLLAMA
# ============================================================
st.sidebar.header("Configuração dos modelos")
# No Streamlit Cloud, configure OLLAMA_BASE_URL em Settings > Secrets.
# Localmente, a variável de ambiente ou o endereço padrão pode ser utilizado.
def get_setting(name, default):
    try:
        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass
    return os.getenv(name, default)


ollama_url = st.sidebar.text_input(
    "Endereço do Ollama",
    value=get_setting("OLLAMA_BASE_URL", "http://localhost:11434"),
    help="Local: http://localhost:11434. No Streamlit Cloud, configure OLLAMA_BASE_URL em Settings > Secrets com o endereço HTTPS do servidor Ollama.",
).strip().rstrip("/")

chat_model = st.sidebar.text_input(
    "Modelo de linguagem",
    value=get_setting("OLLAMA_CHAT_MODEL", "qwen2.5:7b"),
    help="Exemplo: qwen2.5:7b. Baixe o modelo no Ollama antes de executar.",
).strip()

embedding_model = st.sidebar.text_input(
    "Modelo de embeddings",
    value=get_setting("OLLAMA_EMBED_MODEL", "bge-m3"),
    help="Exemplo: bge-m3, modelo multilíngue para gerar vetores de texto.",
).strip()

similarity_threshold = st.sidebar.slider(
    "Similaridade mínima para sugerir agrupamento",
    min_value=0.70,
    max_value=0.99,
    value=0.90,
    step=0.01,
)
review_threshold = st.sidebar.slider(
    "Abaixo deste valor, manter como conceito separado",
    min_value=0.40,
    max_value=0.89,
    value=0.78,
    step=0.01,
)

if review_threshold >= similarity_threshold:
    st.sidebar.warning("O limite de revisão deve ser menor que o limite de agrupamento.")

st.info(
