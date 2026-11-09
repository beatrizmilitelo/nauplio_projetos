    return candidates[0] if candidates else Path(filename).stem

def extract_authors(text):
    # Heurística simples: não garante identificar corretamente todos os autores.
    lines = [re.sub(r"\\s+", " ", line).strip() for line in text.splitlines()[:35]]
    for i, line in enumerate(lines):
        if re.search(r"\\b(abstract|resumo)\\b", line, re.I):
            possible = lines[max(0, i-3):i]
            possible = [x for x in possible if len(x) > 5 and not re.search(r"^(received|accepted|available online)", x, re.I)]
            if possible:
                return " | ".join(possible[-2:])
    return "Não encontrado"

def extract_variables(text):
    found = {}
    for name, patterns in VARIABLES.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                # Guarda uma pequena janela de contexto para facilitar a conferência.
                start, end = max(0, match.start()-100), min(len(text), match.end()+180)
                context = re.sub(r"\\s+", " ", text[start:end]).strip()
                found[name] = context
                break
    return found

if uploaded_zip and st.button("🚀 Processar PDFs e gerar Excel"):
    with st.spinner("Extraindo texto localmente dos PDFs..."):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())

                with zipfile.ZipFile(zip_path, "r") as zf:
                    # Evita extração de caminhos maliciosos (Zip Slip).
                    for member in zf.infolist():
                        target = os.path.realpath(os.path.join(tmpdir, member.filename))
                        if not target.startswith(os.path.realpath(tmpdir) + os.sep):
                            raise ValueError("O ZIP contém um caminho de arquivo inválido.")
                    zf.extractall(tmpdir)

                pdf_files = []
                for root, dirs, files in os.walk(tmpdir):
                    dirs[:] = [d for d in dirs if d != "__MACOSX"]
                    for filename in files:
                        if filename.lower().endswith(".pdf") and not filename.startswith("."):
                            pdf_files.append(os.path.join(root, filename))

                if not pdf_files:
                    st.error("Nenhum PDF válido foi encontrado no ZIP.")
                else:
                    rows, errors = [], []
                    progress = st.progress(0)
                    for i, pdf_path in enumerate(pdf_files):
                        filename = os.path.basename(pdf_path)
                        try:
                            text = extract_pdf_text(pdf_path)
                            if not text.strip():
                                raise ValueError("PDF sem texto extraível (pode ser digitalizado).")
                            row = {
                                "DOI": extract_doi(text),
                                "Nome do Artigo": extract_title(text, filename),
                                "Autor": extract_authors(text),
                            }
                            row.update(extract_variables(text))
                            rows.append(row)
                        except Exception as e:
                            errors.append(f"{filename}: {e}")
                        progress.progress((i + 1) / len(pdf_files))

                    if rows:
                        df = pd.DataFrame(rows)
                        st.success(f"Extração concluída: {len(rows)} artigo(s).")
                        st.subheader("Prévia dos dados")
                        st.dataframe(df.head(20), use_container_width=True)

                        metadata_cols = ["DOI", "Nome do Artigo", "Autor"]
                        variable_cols = [c for c in df.columns if c not in metadata_cols]
                        threshold = 0.30 * len(df)
                        valid_vars = [
                            c for c in variable_cols
                            if df[c].notna().sum() >= threshold
                        ]
                        df_filtered = df[metadata_cols + valid_vars]

                        st.subheader("Variáveis presentes em pelo menos 30% dos artigos")
                        st.write(
                            f"{len(valid_vars)} de {len(variable_cols)} variáveis "
                            f"foram mantidas (limiar: {threshold:.1f} artigos)."
                        )
                        st.dataframe(df_filtered.head(20), use_container_width=True)

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine="openpyxl") as writer:
                            df_filtered.to_excel(writer, index=False, sheet_name="Banco de Dados")
                            df.to_excel(writer, index=False, sheet_name="Todos os dados")
                        st.download_button(
                            "📥 Baixar Excel",
                            data=output.getvalue(),
                            file_name="banco_de_dados_artigos_filtrado.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    else:
                        st.error("Não foi possível extrair texto de nenhum PDF.")
                    if errors:
                        with st.expander(f"Ver {len(errors)} arquivo(s) com erro"):
                            st.write("\\n".join(errors))
