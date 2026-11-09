
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from docling.document_converter import DocumentConverter


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Náuplio | Extrator Científico",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Náuplio Analytics | Extrator Científico")

st.markdown("""
Extração estruturada de artigos científicos utilizando Docling.

- Processamento local de documentos
- Extração de texto estruturado
- Reconhecimento de tabelas
- Extração de variáveis científicas
- Exportação para Excel
- Sem API de IA generativa paga
""")

st.info(
    "O Docling utiliza modelos locais de processamento documental. "
    "Na primeira execução, pode baixar modelos necessários."
)


# ============================================================
# VARIÁVEIS CIENTÍFICAS
# ============================================================

VARIABLES = {
    "latitude": [r"latitude", r"\blat\b"],
    "longitude": [r"longitude", r"\blon\b", r"\blong\b"],
    "temperatura": [r"temperature", r"temperatura"],
    "salinidade": [r"salinity", r"salinidade"],
    "profundidade": [r"depth", r"profundidade"],
    "comprimento": [r"length", r"comprimento"],
    "peso": [r"weight", r"peso", r"\bmass\b"],
    "abundancia": [r"abundance", r"abundância", r"abundancia"],
    "biomassa": [r"biomass", r"biomassa"],
    "densidade": [r"density", r"densidade"],
    "nivel_trofico": [r"trophic level", r"nível trófico"],
    "oxigenio_dissolvido": [r"dissolved oxygen", r"oxigênio dissolvido"],
    "ph": [r"\bpH\b"],
    "clorofila": [r"chlorophyll", r"clorofila"],
    "turbidez": [r"turbidity", r"turbidez"],
    "precipitacao": [r"precipitation", r"precipitação"],
    "velocidade_corrente": [r"current velocity", r"velocidade da corrente"],
    "riqueza_especies": [r"species richness", r"riqueza de espécies"],
    "diversidade": [r"diversity", r"diversidade"],
    "mortalidade": [r"mortality", r"mortalidade"],
    "sobrevivencia": [r"survival", r"sobrevivência"],
    "captura": [r"catch", r"capture", r"captura"],
    "bycatch": [r"bycatch", r"by-catch"],
    "esforco_pesca": [r"fishing effort", r"esforço de pesca"],
}


# ============================================================
# FUNÇÕES
# ============================================================

def extract_doi(text):
    match = re.search(
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
        text,
        re.IGNORECASE
    )

    return match.group(0).rstrip(".,;:") if match else "Não encontrado"


def extract_title(text, filename):
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]

    for line in lines[:40]:
        if (
            25 <= len(line) <= 250
            and not re.search(
                r"^(abstract|resumo|keywords|introduction)",
                line,
                re.IGNORECASE
            )
        ):
            return line

    return Path(filename).stem


def extract_authors(text):
    lines = text.splitlines()[:60]

    for i, line in enumerate(lines):
        if re.search(r"\babstract\b|\bresumo\b", line, re.I):
            possible = lines[max(0, i - 4):i]

            possible = [
                x.strip()
                for x in possible
                if len(x.strip()) > 5
            ]

            if possible:
                return " | ".join(possible[-3:])

    return "Não encontrado"


def extract_variables(text):
    found = {}

    for variable, patterns in VARIABLES.items():

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 200)

                found[variable] = re.sub(
                    r"\s+",
                    " ",
                    text[start:end]
                ).strip()

                break

    return found


def process_pdf(pdf_path, converter):
    """
    Converte PDF com Docling.
    Extrai texto estruturado e tabelas.
    """

    result = converter.convert(pdf_path)

    document = result.document

    # Texto estruturado: títulos, parágrafos, tabelas e conteúdo textual.
    markdown = document.export_to_markdown()

    # --------------------------------------------------------
    # EXTRAÇÃO DAS TABELAS
    # --------------------------------------------------------

    tables_text = []
    tables_data = []

    for index, table in enumerate(document.tables):

        try:

            dataframe = table.export_to_dataframe(
                doc=document
            )

            tables_data.append({
                "Tabela": index + 1,
                "DataFrame": dataframe
            })

            tables_text.append(
                dataframe.to_string(index=False)
            )

        except Exception as error:

            tables_text.append(
                f"Erro na tabela {index + 1}: {error}"
            )

    # Junta texto e conteúdo das tabelas.
    complete_text = markdown + "\n\n" + "\n\n".join(tables_text)

    # --------------------------------------------------------
    # METADADOS E VARIÁVEIS
    # --------------------------------------------------------

    filename = os.path.basename(pdf_path)

    row = {
        "DOI": extract_doi(complete_text),
        "Nome do Artigo": extract_title(
            markdown,
            filename
        ),
        "Autor": extract_authors(markdown),
        "Quantidade de Tabelas": len(document.tables),
        "Quantidade de Figuras": len(document.pictures),
    }

    row.update(
        extract_variables(complete_text)
    )

    return row, tables_data, markdown


# ============================================================
# UPLOAD
# ============================================================

uploaded_zip = st.file_uploader(
    "📁 Envie o ZIP contendo artigos PDF",
    type=["zip"]
)


# ============================================================
# PROCESSAMENTO
# ============================================================

if uploaded_zip:

    if st.button(
        "🚀 Processar artigos com Docling",
        type="primary"
    ):

        with st.spinner(
            "Inicializando Docling e processando PDFs..."
        ):

            try:

                with tempfile.TemporaryDirectory() as tmpdir:

                    zip_path = os.path.join(
                        tmpdir,
                        "artigos.zip"
                    )

                    with open(zip_path, "wb") as f:
                        f.write(uploaded_zip.getbuffer())

                    # ------------------------------------------------
                    # EXTRAÇÃO SEGURA DO ZIP
                    # ------------------------------------------------

                    with zipfile.ZipFile(zip_path) as zip_ref:

                        for member in zip_ref.infolist():

                            target = os.path.realpath(
                                os.path.join(
                                    tmpdir,
                                    member.filename
                                )
                            )

                            if not target.startswith(
                                os.path.realpath(tmpdir) + os.sep
                            ):

                                raise ValueError(
                                    "ZIP contém caminho inválido."
                                )

                        zip_ref.extractall(tmpdir)

                    # ------------------------------------------------
                    # IDENTIFICAR PDFs
                    # ------------------------------------------------

                    pdf_files = []

                    for root, dirs, files in os.walk(tmpdir):

                        dirs[:] = [
                            d for d in dirs
                            if d != "__MACOSX"
                        ]

                        for filename in files:

                            if (
                                filename.lower().endswith(".pdf")
                                and not filename.startswith(".")
                            ):

                                pdf_files.append(
                                    os.path.join(root, filename)
                                )

                    if not pdf_files:

                        st.error(
                            "Nenhum PDF encontrado no ZIP."
                        )

                    else:

                        st.success(
                            f"{len(pdf_files)} PDFs encontrados."
                        )

                        # Um único converter para todo o lote.
                        converter = DocumentConverter()

                        extracted_data = []
                        errors = []
                        all_tables = []
                        all_markdowns = []

                        progress = st.progress(0)

                        # ------------------------------------------------
                        # PROCESSAR PDFs
                        # ------------------------------------------------

                        for i, pdf_path in enumerate(pdf_files):

                            filename = os.path.basename(pdf_path)

                            try:

                                row, tables, markdown = process_pdf(
                                    pdf_path,
                                    converter
                                )

                                row["Arquivo PDF"] = filename

                                extracted_data.append(row)

                                all_markdowns.append({
                                    "Arquivo": filename,
                                    "Markdown": markdown
                                })

                                for table_info in tables:

                                    table_info["Arquivo"] = filename

                                    all_tables.append(
                                        table_info
                                    )

                            except Exception as error:

                                errors.append(
                                    f"{filename}: {error}"
                                )

                            progress.progress(
                                (i + 1) / len(pdf_files)
                            )

                        # ------------------------------------------------
                        # RESULTADOS
                        # ------------------------------------------------

                        if extracted_data:

                            df = pd.DataFrame(
                                extracted_data
                            )

                            st.subheader(
                                "📊 Dados extraídos"
                            )

                            st.dataframe(
                                df,
                                use_container_width=True
                            )

                            # ------------------------------------------------
                            # FILTRAR VARIÁVEIS
                            # ------------------------------------------------

                            metadata_cols = [
                                "DOI",
                                "Nome do Artigo",
                                "Autor",
                                "Quantidade de Tabelas",
                                "Quantidade de Figuras",
                                "Arquivo PDF"
                            ]

                            variable_cols = [
                                col for col in df.columns
                                if col not in metadata_cols
                            ]

                            threshold = 0.30 * len(df)

                            valid_vars = [
                                col
                                for col in variable_cols
                                if df[col].notna().sum() >= threshold
                            ]

                            df_filtered = df[
                                metadata_cols + valid_vars
                            ]

                            st.subheader(
                                "✨ Banco de dados filtrado"
                            )

                            st.write(
                                f"""
                                Artigos processados: {len(df)}

                                Variáveis identificadas: {len(variable_cols)}

                                Variáveis mantidas: {len(valid_vars)}

                                Frequência mínima: 30%
                                """
                            )

                            st.dataframe(
                                df_filtered,
                                use_container_width=True
                            )

                            # ------------------------------------------------
                            # EXCEL
                            # ------------------------------------------------

                            output = io.BytesIO()

                            with pd.ExcelWriter(
                                output,
                                engine="openpyxl"
                            ) as writer:

                                df_filtered.to_excel(
                                    writer,
                                    index=False,
                                    sheet_name="Banco Filtrado"
                                )

                                df.to_excel(
                                    writer,
                                    index=False,
                                    sheet_name="Todos os Dados"
                                )

                                # Exporta cada tabela extraída
                                # para uma aba individual.

                                for index, table_info in enumerate(
                                    all_tables[:100]
                                ):

                                    table_df = table_info["DataFrame"]

                                    sheet_name = (
                                        f"Tabela_{index + 1}"
                                    )

                                    table_df.to_excel(
                                        writer,
                                        index=False,
                                        sheet_name=sheet_name[:31]
                                    )

                            st.download_button(
                                label="📥 Baixar Excel completo",
                                data=output.getvalue(),
                                file_name="nauplio_banco_artigos.xlsx",
                                mime=(
                                    "application/vnd.openxmlformats-officedocument"
                                    ".spreadsheetml.sheet"
                                )
                            )

                            # ------------------------------------------------
                            # DOWNLOAD MARKDOWN
                            # ------------------------------------------------

                            markdown_zip = io.BytesIO()

                            with zipfile.ZipFile(
                                markdown_zip,
                                "w",
                                zipfile.ZIP_DEFLATED
                            ) as zf:

                                for item in all_markdowns:

                                    safe_name = re.sub(
                                        r"[^a-zA-Z0-9_-]",
                                        "_",
                                        Path(item["Arquivo"]).stem
                                    )

                                    zf.writestr(
                                        f"{safe_name}.md",
                                        item["Markdown"]
                                    )

                            st.download_button(
                                label="📥 Baixar artigos convertidos em Markdown",
                                data=markdown_zip.getvalue(),
                                file_name="artigos_markdown.zip",
                                mime="application/zip"
                            )

                        else:

                            st.error(
                                "Não foi possível processar os artigos."
                            )

                        # ------------------------------------------------
                        # ERROS
                        # ------------------------------------------------

                        if errors:

                            with st.expander(
                                f"⚠️ {len(errors)} erros"
                            ):

                                for error in errors:

                                    st.write(error)

            except Exception as error:

                st.error(
                    f"Erro geral: {error}"
                )


st.divider()

st.caption(
    "Náuplio Analytics | Docling + Python + pandas"
)
