import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
import google.generativeai as genai
import json
import io

st.set_page_config(page_title="Extrator de Variáveis de Artigos para Excel", layout="wide")

st.title("📄 Robô Extrator de Variáveis Científicas para Banco de Dados")
st.markdown("""
Este aplicativo processa um arquivo `.zip` contendo vários artigos em **PDF**. 
Ele lê textos, tabelas e imagens utilizando IA, extrai metadados (DOI, Título, Autores) e **todas as variáveis** citadas nos estudos (garantindo **latitude** e **longitude**), organizando em um grande banco de dados em Excel e filtrando apenas as variáveis que aparecem em pelo menos **30% dos artigos**.
""")

# Barra lateral para configuração da API
st.sidebar.header("Configuração")
api_key = st.sidebar.text_input("Insira sua Chave de API do Google Gemini", type="password")

st.sidebar.markdown("""
*Como obter uma chave:* Acesse o [Google AI Studio](https://aistudio.google.com/) e gere sua chave gratuita.
""")

uploaded_zip = st.file_uploader("Envie o arquivo ZIP contendo os PDFs", type=["zip"])

if uploaded_zip and api_key:
    genai.configure(api_key=api_key)
    
    if st.button("🚀 Processar Artigos e Gerar Banco de Dados"):
        with st.spinner("Processando arquivos ZIP e extraindo dados com IA... Isso pode levar alguns minutos dependendo da quantidade de PDFs."):
            
            # Criar diretório temporário para descompactar os arquivos
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Listar todos os PDFs extraídos (ignorando arquivos ocultos de sistema como __MACOSX)
                pdf_files = []
                for root, dirs, files in os.walk(tmpdir):
                    if "__MACOSX" in root:
                        continue
                    for file in files:
                        if file.lower().endswith('.pdf') and not file.startswith('._'):
                            pdf_files.append(os.path.join(root, file))
                
                if not pdf_files:
                    st.error("Nenhum arquivo PDF válido foi encontrado dentro do arquivo ZIP.")
                else:
                    st.info(f"Encontrados {len(pdf_files)} arquivos PDF. Iniciando extração inteligente...")
                    
                    extracted_data = []
                    progress_bar = st.progress(0)
                    
                    # Configuração para forçar a API a responder em formato JSON estrito
                    generation_config = {
                        "response_mime_type": "application/json"
                    }
                    
                    model = genai.GenerativeModel('gemini-1.5-pro', generation_config=generation_config)
                    
                    for i, pdf_path in enumerate(pdf_files):
                        sample_file = None
                        try:
                            # Fazer upload do PDF para a API do Gemini
                            sample_file = genai.upload_file(pdf_path)
                            
                            prompt = """
                            Analise este artigo científico completo, incluindo textos, tabelas e imagens/figuras.
                            Extraia as seguintes informações e retorne estritamente em formato JSON:
                            
                            1. "doi": O DOI do artigo (string ou "Não encontrado").
                            2. "article_name": O título completo do artigo.
                            3. "authors": Os autores principais do artigo.
                            4. "variables": Um dicionário contendo TODAS as variáveis medidas, coletadas, analisadas ou citadas no estudo (ex: latitude, longitude, comprimento máximo, nível trófico, temperatura, salinidade, etc.), junto com seus respectivos valores ou menções encontradas. As chaves devem ser o nome padronizado da variável (em minúsculas e sem acentos complexos, ex: "latitude", "longitude", "comprimento_maximo") e o valor deve ser o dado extraído ou "Presente".
                            
                            Formato esperado:
                            {
                              "doi": "...",
                              "article_name": "...",
                              "authors": "...",
                              "variables": {
                                "latitude": "-23.55",
                                "longitude": "-46.63",
                                "outra_variavel": "valor"
                              }
                            }
                            """
                            
                            response = model.generate_content([sample_file, prompt])
                            
                            # Parse direto do JSON (garantido pelo response_mime_type)
                            data = json.loads(response.text.strip())
                            
                            # Montar linha do DataFrame
                            row = {
                                "DOI": data.get("doi", "Não encontrado"),
                                "Nome do Artigo": data.get("article_name", os.path.basename(pdf_path)),
                                "Autor": data.get("authors", "Não encontrado")
                            }
                            
                            vars_dict = data.get("variables", {})
                            if isinstance(vars_dict, dict):
                                for k, v in vars_dict.items():
                                    row[str(k).strip().lower()] = v
                                    
                            extracted_data.append(row)
                            
                        except Exception as e:
                            st.warning(f"Erro ao processar o arquivo {os.path.basename(pdf_path)}: {str(e)}")
                            
                        finally:
                            # Garante a eliminação do ficheiro temporário na cloud, mesmo que ocorra um erro
                            if sample_file:
                                try:
                                    genai.delete_file(sample_file.name)
                                except Exception:
                                    pass
                        
                        progress_bar.progress((i + 1) / len(pdf_files))
                    
                    if extracted_data:
                        df = pd.DataFrame(extracted_data)
                        
                        st.success("Extração concluída com sucesso!")
                        st.subheader("📊 Prévia dos Dados Brutos Extraídos")
                        st.dataframe(df.head())
                        
                        # Filtrar variáveis que aparecem em pelo menos 30% dos artigos
                        total_articles = len(df)
                        threshold = 0.3 * total_articles
                        
                        metadata_cols = ["DOI", "Nome do Artigo", "Autor"]
                        variable_cols = [col for col in df.columns if col not in metadata_cols]
                        
                        # Normalizar valores ausentes/inválidos para um cálculo exato de ocorrência
                        df_vars = df[variable_cols].replace(["Não encontrado", "N/A", "n/a", "", None], pd.NA)
                        
                        valid_vars = []
                        for col in variable_cols:
                            non_null_count = df_vars[col].notna().sum()
                            if non_null_count >= threshold:
                                valid_vars.append(col)
                                
                        final_cols = metadata_cols + valid_vars
                        df_filtered = df[[col for col in final_cols if col in df.columns]]
                        
                        st.subheader(f"✨ Dados Filtrados (Variáveis presentes em ≥ 30% dos {total_articles} artigos)")
                        st.write(f"De **{len(variable_cols)}** variáveis totais extraídas, **{len(valid_vars)}** atenderam ao critério de frequência (aparecem em 30% ou mais dos artigos) e foram mantidas. As demais foram descartadas.")
                        st.dataframe(df_filtered.head())
                        
                        # Gerar arquivo Excel para download
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_filtered.to_excel(writer, index=False, sheet_name='Banco de Dados')
                        processed_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 Baixar Planilha Excel Final Consolidada",
                            data=processed_data,
                            file_name="banco_de_dados_artigos_filtrado.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("Nenhum dado pôde ser extraído dos arquivos PDF.")

elif uploaded_zip and not api_key:
    st.warning("⚠️ Por favor, insira sua Chave de API do Google Gemini na barra lateral para continuar.")
