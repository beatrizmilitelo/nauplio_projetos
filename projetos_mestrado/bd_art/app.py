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
Este sistema extrai a estrutura dos PDFs com **Docling**, utiliza o **Ollama** local para identificar 
metadados (Título, Autores, DOI) e variáveis científicas sem um dicionário fixo, e consolida 
tudo em uma planilha Excel.
"""
)

# ============================================================
# CONFIGURAÇÃO DO OLLAMA
# ============================================================
st.sidebar.header("Configuração dos modelos")

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
).strip().rstrip("/")

chat_model = st.sidebar.text_input(
    "Modelo de linguagem",
    value=get_setting("OLLAMA_CHAT_MODEL", "qwen2.5:7b"),
).strip()

st.info("PDFs precisam ter texto selecionável. A extração é processada localmente via Ollama.")

# ============================================================
# INICIALIZAÇÃO DO DOCLING
# ============================================================
@st.cache_resource(show_spinner="Inicializando Docling...")
def load_docling_converter():
    options = PdfPipelineOptions()
    options.do_ocr = False  # Desativado para evitar dependências de OCR
    options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

converter = load_docling_converter()

# ============================================================
# FUNÇÃO DE EXTRAÇÃO COM OLLAMA
# ============================================================
def extract_metadata_and_variables(text_content, model_name, api_url):
    """
    Envia o texto do artigo para o Ollama e solicita resposta em formato JSON contendo
    Título, Autores, DOI e as variáveis encontradas.
    """
    prompt = f"""
    Analise o texto acadêmico a seguir e extraia as seguintes informações em formato JSON estrito:
    1. "titulo": Título do artigo.
    2. "autores": Nome dos autores (separados por vírgula).
    3. "doi": Código DOI do artigo (se presente, caso contrário null).
    4. "variaveis": Um dicionário/objeto com as variáveis científicas, metodológicas ou conceituais identificadas no artigo e seus respectivos valores ou descrições (chave: nome da variável, valor: descrição/valor).

    Exemplo de formato de resposta:
    {{
        "titulo": "Exemplo de Título",
        "autores": "Silva A., Souza B.",
        "doi": "10.1016/j.example.2023.01.001",
        "variaveis": {{
            "Temperatura da água": "25°C",
            "Espécie analisada": "Litopenaeus vannamei",
            "Salinidade": "35 ppt"
        }}
    }}

    Texto do artigo (primeiros trechos):
    {text_content[:8000]}
    """

    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }

    try:
        response = requests.post(f"{api_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        return json.loads(res_json.get("response", "{}"))
    except Exception as e:
        st.error(f"Erro na comunicação com Ollama: {e}")
        return {"titulo": None, "autores": None, "doi": None, "variaveis": {}}

# ============================================================
# CARREGAMENTO E PROCESSAMENTO DO ARQUIVO ZIP
# ============================================================
st.header("📁 Processamento em Lote (.zip)")

uploaded_zip = st.file_uploader(
    "Carregue um arquivo ZIP contendo os PDFs",
    type=["zip"],
    help="O arquivo .zip deve conter os artigos acadêmicos em formato PDF."
)

if uploaded_zip is not None:
    # 1. Leitura do arquivo ZIP
    pdf_files = []
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        for filename in z.namelist():
            if filename.lower().endswith(".pdf") and not filename.startswith("__MACOSX"):
                pdf_data = z.read(filename)
                pdf_io = io.BytesIO(pdf_data)
                pdf_io.name = os.path.basename(filename)
                pdf_files.append(pdf_io)

    st.success(f"Encontrados **{len(pdf_files)}** arquivos PDF.")

    if st.button("🚀 Iniciar Extração Semântica", type="primary"):
        all_records = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processando arquivo ({idx+1}/{len(pdf_files)}): {pdf_file.name}...")
            
            # Salva temporariamente em memória/disco para leitura pelo Docling
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            try:
                # 2. Convertendo PDF com Docling
                conversion_result = converter.convert(tmp_path)
                markdown_text = conversion_result.document.export_to_markdown()

                # 3. Extraindo informações via Ollama
                extracted_data = extract_metadata_and_variables(
                    markdown_text, chat_model, ollama_url
                )

                # 4. Montagem da linha do DataFrame
                row = {
                    "Arquivo PDF": pdf_file.name,
                    "Título": extracted_data.get("titulo", "N/A"),
                    "Autores": extracted_data.get("autores", "N/A"),
                    "DOI": extracted_data.get("doi", "N/A"),
                }

                # Adiciona cada variável identificada como uma coluna
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

        status_text.text("Processamento concluído!")

        # 5. Geração do DataFrame e Botão de Exportação para Excel
        if all_records:
            df_result = pd.DataFrame(all_records)
            
            # Reorganiza colunas fixas no início
            fixed_cols = ["Arquivo PDF", "Título", "Autores", "DOI"]
            other_cols = [c for c in df_result.columns if c not in fixed_cols]
            df_result = df_result[fixed_cols + other_cols]

            st.subheader("📊 Tabela Extraída")
            st.dataframe(df_result, use_container_width=True)

            # Conversão para buffer Excel
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
