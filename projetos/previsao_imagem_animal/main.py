# =========================================================
# CORRESPONDÊNCIA DO TEXTO COM AS PREVISÕES
# =========================================================

def verificar_correspondencia(termo, rotulo):
    termo_normalizado = normalizar(termo)
    rotulo_normalizado = normalizar(rotulo)

    if not termo_normalizado or not rotulo_normalizado:
        return False

    if termo_normalizado == rotulo_normalizado:
        return True

    if termo_normalizado in rotulo_normalizado or rotulo_normalizado in termo_normalizado:
        return True

    palavras_termo = set(termo_normalizado.split())
    palavras_rotulo = set(rotulo_normalizado.split())

    # Ignora palavras muito comuns/pequenas que poluem a busca (opcional)
    stopwords = {"a", "o", "de", "do", "da", "em", "um", "uma"}
    palavras_termo = {p for p in palavras_termo if p not in stopwords and len(p) >= 2}
    palavras_rotulo = {p for p in palavras_rotulo if p not in stopwords and len(p) >= 2}

    # Mapeamento rápido de sinônimos comuns em termos compostos (ex: mobile/cellular)
    sinonimos = {
        "mobile": "cellular",
        "cellular": "mobile",
        "phone": "telephone",
        "telephone": "phone"
    }

    # Expande as palavras do termo com sinônimos conhecidos
    palavras_termo_expandidas = set(palavras_termo)
    for p in palavras_termo:
        if p in sinonimos:
            palavras_termo_expandidas.add(sinonimos[p])

    if palavras_termo_expandidas.intersection(palavras_rotulo):
        return True

    return False
