import io
import os
import re
import json
import zipfile
import tempfile
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
Este sistema extrai a estrutura dos PDFs com **Docling**, utiliza o **Ollama** local/remoto 
para identificar metadados (Título, Autores, DOI) e variáveis científicas sem um dicionário fixo, 
e consolida tudo em uma planilha Excel pronta para download.
"""
)


# ============================================================
# CONFIGURAÇÃO DO OLLAMA (Local ou Streamlit Cloud)
# ============================================================
st.sidebar.header("⚙️ Configuração dos modelos")

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
    help="No Streamlit Cloud, use uma URL pública (ex: Ngrok: https://xxxx.ngrok-free.app). Localmente: http://localhost:11434",
).strip().rstrip("/")

chat_model = st.sidebar.text_input(
    "Modelo de linguagem",
    value=get_setting("OLLAMA_CHAT_MODEL", "qwen2.5:7b"),
    help="Modelo do Ollama instalado (ex: qwen2.5:7b, llama3, mistral).",
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
    "ℹ️ **Dica para Streamlit Cloud:** Como o app está rodando na nuvem, você precisa "
    "informar uma URL acessível publicamente no campo **Endereço do Ollama** (ex: via Ngrok) "
    "ou hospedar o Ollama em um servidor remoto."
)


# ============================================================
# DOCLING PIPELINE SETUP
# ============================================================
@st.cache_resource(show_spinner="Inicializando Docling...")
def load_docling_converter():
    options = PdfPipelineOptions()
    options.do_ocr = False  # OCR desativado para otimização e evitar erro de permissão do RapidOCR
    options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

converter = load_docling_converter()


# ============================================================
# EXTRAÇÃO VIA OLLAMA (LLM)
# ============================================================
def extract_metadata_and_variables(text_content, model_name, api_url):
    """
    Envia o texto do artigo para o Ollama e retorna JSON estrito
    contendo Título, Autores, DOI e um dicionário de variáveis.
    """
    prompt = f"""
    Analise o texto acadêmico a seguir e extraia as seguintes informações em formato JSON estrito:
    1. "titulo": Título do artigo.
    2. "autores": Nome dos autores (separados por vírgula).
    3. "doi": Código DOI do artigo (se presente, caso contrário null).
    4. "variaveis": Um dicionário/objeto com as variáveis científicas, metodológicas ou conceituais identificadas no artigo e seus respectivos valores ou descrições (chave: nome da variável, valor: descrição/valor).

    Exemplo de formato de resposta esperada:
    {{
        "titulo": "Exemplo de Título Acadêmico",
        "autores": "Silva A., Souza B.",
        "doi": "10.1016/j.example.2023.01.001",
        "variaveis": {{
            "Temperatura da água": "25°C",
            "Espécie analisada": "Litopenaeus vannamei",
            "Salinidade": "35 ppt"
        }}
    }}

    Texto do artigo (primeiros caracteres):
    {text_content[:8000]}
    """

    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    try:
        url = f"{api_url}/api/generate"
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        res_json = response.json()
        
        raw_output = res_json.get("response", "{}")
        return json.loads(raw_output)

    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ **Erro de Conexão com Ollama** em `{api_url}`.\n\n"
            "Se o app está no Streamlit Cloud, certifique-se de que a URL no menu lateral é "
            "uma URL pública ativa (ex: `https://xxxx.ngrok-free.app`) e não `localhost`."
        )
        return {"titulo": None, "autores": None, "doi": None, "variaveis": {}}
    except Exception as e:
        st.error(f"⚠️ Erro ao processar resposta do Ollama: {e}")
        return {"titulo": None, "autores": None, "doi": None, "variaveis": {}}


# ============================================================
# CARREGAMENTO E PROCESSAMENTO DO ARQUIVO ZIP
# ============================================================
st.header("📁 Upload do Arquivo ZIP contendo os PDFs")

uploaded_zip = st.file_uploader(
    "Carregue um arquivo .zip com os PDFs dos artigos",
    type=["zip"],
    help="O arquivo .zip deve conter os PDFs a serem processados."
)

if uploaded_zip is not None:
    # Descompacta PDFs em memória
    pdf_files = []
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        for filename in z.namelist():
            if filename.lower().endswith(".pdf") and not filename.startswith("__MACOSX"):
                pdf_data = z.read(filename)
                pdf_io = io.BytesIO(pdf_data)
                pdf_io.name = os.path.basename(filename)
                pdf_files.append(pdf_io)

    st.success(f"Encontrados **{len(pdf_files)}** arquivos PDF no arquivo ZIP.")

    if st.button("🚀 Iniciar Extração Semântica", type="primary"):
        all_records = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processando ({idx + 1}/{len(pdf_files)}): {pdf_file.name}...")
            
            # Escreve arquivo temporário para leitura do Docling
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            try:
                # 1. Extração de texto com Docling
                conversion_result = converter.convert(tmp_path)
                markdown_text = conversion_result.document.export_to_markdown()

                # 2. Extração semântica via Ollama
                extracted_data = extract_metadata_and_variables(
                    markdown_text, chat_model, ollama_url
                )

                # 3. Montagem da linha de registro
                row = {
                    "Arquivo PDF": pdf_file.name,
                    "Título": extracted_data.get("titulo", "N/A"),
                    "Autores": extracted_data.get("autores", "N/A"),
                    "DOI": extracted_data.get("doi", "N/A"),
                }

                # Expande dicionário de variáveis como colunas individuais
                variables = extracted_data.get("variaveis", {})
                if isinstance(variables, dict):
                    for var_name, var_value in variables.items():
                        row[var_name] = str(var_value)

                all_records.append(row)

            except Exception as err:
                st.warning(f"Falha ao processar `{pdf_file.name}`: {err}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            progress_bar.progress((idx + 1) / len(pdf_files))

        status_text.text("✨ Processamento concluído com sucesso!")

        # 4. Exibição e exportação da planilha Excel
        if all_records:
            df_result = pd.DataFrame(all_records)
            
            # Ordena colunas base no início da planilha
            fixed_cols = ["Arquivo PDF", "Título", "Autores", "DOI"]
            other_cols = [c for c in df_result.columns if c not in fixed_cols]
            df_result = df_result[fixed_cols + other_cols]

            st.subheader("📊 Tabela Extraída")
            st.dataframe(df_result, use_container_width=True)

            # Gravação da planilha em memória
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_result.to_excel(writer, index=False, sheet_name="Artigos_Extraidos")
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Baixar Resultado em Excel (.xlsx)",
                data=excel_buffer,
                file_name="extracao_artigos_nauplio.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
