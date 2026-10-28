import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import os

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram

# ============================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Análise Semântica de Abstracts",
    page_icon="🔬",
    layout="wide"
)

# Ocultar warnings desnecessários
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16

# ============================================================
# CACHE DO MODELO (Evita recarregar o modelo pesado a cada clique)
# ============================================================
@st.cache_resource
def load_model():
    with st.spinner(f"Carregando o modelo de IA ({MODEL_NAME}). Isso pode levar alguns segundos na primeira execução..."):
        model = SentenceTransformer(MODEL_NAME)
    return model

# ============================================================
# INTERFACE LATERAL (Sidebar)
# ============================================================
st.sidebar.title("🔬 Configurações")
st.sidebar.markdown("Faça o upload dos seus arquivos CSV para iniciar a análise.")

uploaded_abs = st.sidebar.file_uploader("Arquivo de Abstracts (`abstracts.csv`)", type=["csv"])
uploaded_desc = st.sidebar.file_uploader("Arquivo de Descritores (`descriptions_invented.csv`)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros")
top_n_similar = st.sidebar.slider("Artigos semelhantes (Top N)", 2, 10, 5)
top_n_desc = st.sidebar.slider("Descritores por abstract", 1, 10, 5)
min_clusters = st.sidebar.slider("Min Clusters (KMeans)", 2, 5, 2)
max_clusters = st.sidebar.slider("Max Clusters (KMeans)", 6, 15, 10)

# ============================================================
# CORPO PRINCIPAL
# ============================================================
st.title("Exploração Semântica de Abstracts Científicos")
st.markdown("""
Esta aplicação utiliza modelos de linguagem avançados (`BAAI/bge-m3`) para realizar análises semânticas, 
agrupamentos e correlações entre artigos científicos e descritores/conceitos.
""")

if uploaded_abs is not None and uploaded_desc is not None:
    # Leitura dos dados enviados
    try:
        df_abs = pd.read_csv(uploaded_abs)
        df_desc = pd.read_csv(uploaded_desc)
    except Exception as e:
        st.error(fempre="Erro ao ler os arquivos CSV: {e}")
        st.stop()

    # Validações básicas
    if "abstract" not in df_abs.columns:
        st.error("O arquivo de abstracts precisa conter uma coluna chamada `abstract`.")
        st.stop()
    if "description_id" not in df_desc.columns or "description_text" not in df_desc.columns:
        st.error("O arquivo de descritores precisa conter as colunas `description_id` e `description_text`.")
        st.stop()

    # Pré-processamento rápido
    df_abs = df_abs.dropna(subset=["abstract"]).copy()
    df_abs["abstract"] = df_abs["abstract"].astype(str).str.strip()
    df_abs = df_abs.reset_index(names="source_row_id")
    
    df_desc = df_desc.dropna(subset=["description_text"]).copy()
    df_desc["description_text"] = df_desc["description_text"].astype(str).str.strip()
    df_desc = df_desc.reset_index(drop=True)

    st.success(f"Dados carregados com sucesso! **{len(df_abs)}** abstracts e **{len(df_desc)}** descritores encontrados.")

    # Botão para rodar o pipeline
    if st.button("Executar Análise Completa", type="primary"):
        model = load_model()

        with st.spinner("Gerando embeddings semânticos..."):
            abstracts = df_abs["abstract"].tolist()
            descriptions = df_desc["description_text"].tolist()

            abstract_embeddings = model.encode(
                abstracts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
            )
            description_embeddings = model.encode(
                descriptions, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
            )

        # Matriz de similaridade
        abstract_descriptor_matrix = abstract_embeddings @ description_embeddings.T
        abstract_abstract_matrix = abstract_embeddings @ abstract_embeddings.T
        
        descriptor_ids = df_desc["description_id"].tolist()
        abstract_ids = df_abs["source_row_id"].tolist()

        similarity_df = pd.DataFrame(abstract_descriptor_matrix, columns=descriptor_ids)
        similarity_df.insert(0, "abstract_id", abstract_ids)

        # Salvando no session_state para exibir nas abas
        st.session_state["run_success"] = True
        st.session_state["df_abs"] = df_abs
        st.session_state["df_desc"] = df_desc
        st.session_state["similarity_df"] = similarity_df
        st.session_state["abstract_embeddings"] = abstract_embeddings
        st.session_state["abstract_abstract_matrix"] = abstract_abstract_matrix
        st.session_state["descriptor_ids"] = descriptor_ids

    # Exibição dos resultados se já foram processados
    if st.session_state.get("run_success", False):
        df_abs = st.session_state["df_abs"]
        df_desc = st.session_state["df_desc"]
        similarity_df = st.session_state["similarity_df"]
        abstract_embeddings = st.session_state["abstract_embeddings"]
        abstract_abstract_matrix = st.session_state["abstract_abstract_matrix"]
        descriptor_ids = st.session_state["descriptor_ids"]

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔗 Artigos Semelhantes", 
            "📊 Descritores por Abstract", 
            "🎯 Clusters (KMeans)", 
            "🔥 Heatmap & Dendrograma"
        ])

        # ABA 1: Artigos Semelhantes
        with tab1:
            st.subheader("Artigos Mais Semelhantes (Vizinhos Semânticos)")
            article_ids = df_abs["source_row_id"].tolist()
            title_col = "title" if "title" in df_abs.columns else "source_row_id"
            
            rows = []
            for i in range(len(df_abs)):
                scores = abstract_abstract_matrix[i].copy()
                scores[i] = -np.inf
                top_indices = np.argsort(scores)[::-1][:top_n_similar]
                for rank, j in enumerate(top_indices, start=1):
                    rows.append({
                        "ID Abstract": article_ids[i],
                        "Título": df_abs.iloc[i].get(title_col, f"Abstract {article_ids[i]}"),
                        "Ranking": rank,
                        "ID Similar": article_ids[j],
                        "Título Similar": df_abs.iloc[j].get(title_col, f"Abstract {article_ids[j]}"),
                        "Similaridade de Cosseno": float(scores[j])
                    })
            df_sim_res = pd.DataFrame(rows)
            st.dataframe(df_sim_res, use_container_width=True)

        # ABA 2: Descritores por Abstract
        with tab2:
            st.subheader("Rankings de Descritores")
            desc_lookup = df_desc.set_index("description_id")["description_text"].to_dict()
            title_lookup = df_abs.set_index("source_row_id").get("title", pd.Series(dtype=object)).to_dict()

            rows_by_article = []
            for _, row in similarity_df.iterrows():
                ab_id = row["abstract_id"]
                scores = row[descriptor_ids].sort_values(ascending=False).head(top_n_desc)
                for rank, (desc_id, score) in enumerate(scores.items(), start=1):
                    rows_by_article.append({
                        "abstract_id": ab_id,
                        "Título": title_lookup.get(ab_id, f"Abstract {ab_id}"),
                        "Rank": rank,
                        "ID Descritor": desc_id,
                        "Texto Descritor": desc_lookup.get(desc_id, ""),
                        "Similaridade": float(score)
                    })
            st.dataframe(pd.DataFrame(rows_by_article), use_container_width=True)

        # ABA 3: Clusters
        with tab3:
            st.subheader("Análise de Clusters (KMeans)")
            n_articles = len(df_abs)
            if n_articles >= 3:
                max_k = min(max_clusters, n_articles - 1)
                min_k = min(min_clusters, max_k)
                
                best_sil = -1
                best_k = min_k
                best_labels = None
                best_model = None

                metric_rows = []
                for k in range(min_k, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(abstract_embeddings)
                    if len(np.unique(labels)) > 1:
                        sil = silhouette_score(abstract_embeddings, labels, metric="euclidean")
                        metric_rows.append({"Clusters": k, "Silhouette": sil})
                        if sil > best_sil:
                            best_sil = sil
                            best_k = k
                            best_labels = labels
                            best_model = kmeans

                st.write(f"**Melhor número de clusters encontrado (por Silhouette):** k = {best_k} (Score: {best_sil:.4f})")
                
                # Gráfico PCA dos clusters
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(abstract_embeddings)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=best_labels, cmap="tab10", s=50, alpha=0.8)
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                ax.set_title(f"Visualização PCA dos Clusters (k={best_k})")
                plt.colorbar(scatter, label="Cluster")
                st.pyplot(fig)
            else:
                st.warning("São necessários pelo menos 3 artigos para realizar o clustering.")

        # ABA 4: Heatmap e Dendrograma
        with tab4:
            st.subheader("Heatmap de Similaridade (Abstract x Descritor)")
            matrix_plot = similarity_df.set_index("abstract_id")[descriptor_ids]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(matrix_plot, cmap="viridis", ax=ax, cbar=True)
            ax.set_xlabel("ID do Descritor")
            ax.set_ylabel("ID do Abstract")
            st.pyplot(fig)

            if len(df_abs) <= 200 and len(df_abs) >= 3:
                st.subheader("Dendrograma Hierárquico")
                Z = linkage(abstract_embeddings, method="average", metric="cosine")
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, leaf_rotation=90, leaf_font_size=8, ax=ax)
                ax.set_title("Agrupamento Hierárquico (Distância Cosseno)")
                st.pyplot(fig)

else:
    st.info("👈 Por favor, faça o upload dos arquivos `abstracts.csv` e `descriptions_invented.csv` na barra lateral para começar.")
