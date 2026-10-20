
TRADUCOES_CONHECIDAS = {
    # Animais
    "porco": "pig",
    "porca": "sow",
    "leitão": "piglet",
    "leitao": "piglet",
    "tartaruga": "turtle",
    "cachorro": "dog",
    "cão": "dog",
    "cao": "dog",
    "gato": "cat",
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
    "leão": "lion",
    "leao": "lion",
    "tigre": "tiger",
    "urso": "bear",
    "cobra": "snake",
    "peixe": "fish",
    "golfinho": "dolphin",
    "baleia": "whale",
    "tubarão": "shark",
    "tubarao": "shark",
    "papagaio": "parrot",
    "borboleta": "butterfly",

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
    "relógio": "clock",
    "relogio": "clock",
    "óculos": "glasses",
    "oculos": "glasses",
}


@st.cache_data(ttl=86400, show_spinner=False)
def traduzir_para_ingles(texto, idioma_origem):
    texto = texto.strip()

    if not texto:
        raise ValueError("Digite o nome de um animal ou objeto.")

    # Para português, prioriza traduções conhecidas e verificadas.
    if idioma_origem == "pt-BR":
        chave = normalizar(texto)

        if chave in TRADUCOES_CONHECIDAS:
            return TRADUCOES_CONHECIDAS[chave]

    erros = []

    try:
        resultado = GoogleTranslator(
            source="auto" if idioma_origem == "auto" else idioma_origem,
            target="en",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip()

    except Exception as erro:
        erros.append(f"Google: {str(erro)[:200]}")

    try:
        origem = "pt" if idioma_origem == "pt-BR" else idioma_origem

        resultado = MyMemoryTranslator(
            source=origem,
            target="en-GB",
        ).translate(texto)

        if resultado and resultado.strip():
            return resultado.strip()

    except Exception as erro:
        erros.append(f"MyMemory: {str(erro)[:200]}")

    raise RuntimeError(
        "Não foi possível traduzir o texto. "
        "Confira o idioma selecionado ou tente novamente mais tarde. "
        + " | ".join(erros)
    )
