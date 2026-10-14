
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Robô de Envio de E-mails</h1>"

import io
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import parseaddr
from html import escape

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Robô de Envio de E-mails</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                padding: 20px;
            }
            .container {
                max-width: 600px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                margin: auto;
            }
            h2 { color: #333; }
            label {
                font-weight: bold;
                display: block;
                margin-top: 10px;
            }
            input, textarea {
                width: 100%;
                padding: 8px;
                margin-top: 5px;
                box-sizing: border-box;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                background: #007BFF;
                color: white;
                border: none;
                padding: 10px 15px;
                margin-top: 20px;
                cursor: pointer;
                width: 100%;
                border-radius: 4px;
                font-size: 16px;
            }
            button:hover { background: #0056b3; }
            .info {
                background: #e7f3fe;
                border-left: 6px solid #2196F3;
                padding: 10px;
                margin-top: 15px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📧 Robô de Envio de E-mails</h2>
            <p>Envie e-mails personalizados utilizando uma conta Gmail.</p>
            <hr>

            <form action="/enviar" method="post"
                  enctype="multipart/form-data">

                <h3>🔐 Dados do Remetente</h3>

                <label>E-mail do Gmail:</label>
                <input type="email" name="sender_email" required>

                <label>Senha de Aplicativo do Gmail:</label>
                <input type="password" name="sender_password"
                       autocomplete="new-password" required>

                <div class="info">
                    A senha utilizada não é a senha normal da conta Google.
                    É necessário criar uma Senha de Aplicativo em uma conta
                    Google elegível com a verificação em duas etapas ativada.
                    <br>
                    <a href="https://myaccount.google.com/apppasswords"
                       target="_blank" rel="noopener noreferrer">
                       Criar Senha de Aplicativo
                    </a>
                </div>

                <h3>📄 Planilha de Destinatários</h3>
                <label>Arquivo Excel (.xlsx) com uma coluna chamada Email:</label>
                <input type="file" name="excel_file"
                       accept=".xlsx" required>

                <h3>🖼 Imagem</h3>
                <label>Imagem do corpo do e-mail (.png, .jpg ou .jpeg):</label>
                <input type="file" name="image_file"
                       accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                       required>

                <h3>✉ Conteúdo</h3>

                <label>Assunto:</label>
                <input type="text" name="subject"
                       value="Campanha Promocional" required>

                <label>HTML do E-mail:</label>
                <textarea name="body_html" rows="12" required><html>
<body>
    <h1>Promoção Especial!</h1>
    <p>Confira nossa promoção exclusiva.</p>
    <img src="cid:image1" alt="Imagem da campanha"
         style="max-width:100%; height:auto;">
    <p>Esperamos você!</p>
</body>
</html></textarea>

                <button type="submit">🚀 Enviar E-mails</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/enviar", response_class=HTMLResponse)
async def enviar_emails(
    sender_email: str = Form(...),
    sender_password: str = Form(...),
    excel_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    subject: str = Form(...),
    body_html: str = Form(...),
):
    server = None

    try:
        # 1. Validar o remetente
        sender_email = sender_email.strip()
        nome, endereco = parseaddr(sender_email)

        if not endereco or "@" not in endereco:
            return pagina_resultado(
                "Erro",
                "O e-mail do remetente é inválido."
            )

        # 2. Ler a planilha Excel usando BytesIO
        contents_excel = await excel_file.read()

        if not contents_excel:
            return pagina_resultado(
                "Erro",
                "A planilha enviada está vazia."
            )

        df = pd.read_excel(io.BytesIO(contents_excel), engine="openpyxl")

        # Identificar a coluna Email sem diferenciar maiúsculas/minúsculas
        colunas = {
            str(coluna).strip().lower(): coluna
            for coluna in df.columns
        }

        if "email" not in colunas:
            return pagina_resultado(
                "Erro",
                "A planilha precisa possuir uma coluna chamada Email."
            )

        coluna_email = colunas["email"]

        # Remover células vazias, espaços e duplicatas
        recipients = (
            df[coluna_email]
            .dropna()
            .astype(str)
            .str.strip()
        )

        recipients = list(dict.fromkeys(
            email for email in recipients
            if email and email.lower() not in ("nan", "none")
        ))

        if not recipients:
            return pagina_resultado(
                "Erro",
                "Nenhum destinatário foi encontrado na planilha."
            )

        # 3. Validar e ler a imagem
        image_bytes = await image_file.read()

        if not image_bytes:
            return pagina_resultado(
                "Erro",
                "O arquivo de imagem está vazio."
            )

        tipo_imagem = imghdr.what(None, h=image_bytes)

        if tipo_imagem not in ("jpeg", "png"):
            return pagina_resultado(
                "Erro",
                "A imagem deve ser um arquivo PNG ou JPEG válido."
            )

        # 4. Preparar o conteúdo do e-mail
        subject = subject.strip()

        if not subject:
            return pagina_resultado(
                "Erro",
                "O assunto do e-mail não pode ficar vazio."
            )

        # 5. Conectar ao Gmail
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, sender_password.strip())

        enviados = 0
        falhas = []

        # 6. Enviar individualmente
        for recipient in recipients:
            nome_destinatario, endereco_destinatario = parseaddr(recipient)

            if (
                not endereco_destinatario
                or "@" not in endereco_destinatario
                or endereco_destinatario != recipient
            ):
                falhas.append((recipient, "Endereço de e-mail inválido"))
                continue

            try:
                msg = MIMEMultipart("related")
                msg["From"] = sender_email
                msg["To"] = endereco_destinatario
                msg["Subject"] = subject

                corpo = MIMEMultipart("alternative")
                corpo.attach(
                    MIMEText(body_html, "html", "utf-8")
                )
                msg.attach(corpo)

                subtipo = "png" if tipo_imagem == "png" else "jpeg"
                img = MIMEImage(image_bytes, _subtype=subtipo)
                img.add_header("Content-ID", "<image1>")
                img.add_header(
                    "Content-Disposition",
                    "inline",
                    filename=f"imagem.{subtipo}"
                )
                msg.attach(img)

                recusados = server.sendmail(
                    sender_email,
                    [endereco_destinatario],
                    msg.as_string()
                )

                if recusados:
                    falhas.append((
                        recipient,
                        "Servidor recusou o destinatário"
                    ))
                else:
                    enviados += 1

            except Exception as erro_email:
                logger.exception(
                    "Falha ao enviar e-mail para %s",
                    recipient
                )
                falhas.append((
                    recipient,
                    str(erro_email)[:250]
                ))

        return pagina_resultado(
            "Resultado do envio",
            f"Processamento concluído. Enviados: {enviados}. "
            f"Falhas: {len(falhas)}. "
            f"Total de destinatários: {len(recipients)}.",
            falhas=falhas
        )

    except Exception as erro:
        logger.exception("Erro no processamento dos e-mails")
        return pagina_resultado(
            "Erro ao processar",
            f"{type(erro).__name__}: {str(erro)[:500]}"
        )

    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass


def pagina_resultado(titulo, mensagem, falhas=None):
    titulo_html = escape(str(titulo))
    mensagem_html = escape(str(mensagem))

    html_falhas = ""

    if falhas:
        itens = "".join(
            f"<li><b>{escape(email)}</b>: {escape(erro)}</li>"
            for email, erro in falhas[:30]
        )
        html_falhas = (
            "<h3>Detalhes das falhas (até 30)</h3>"
            f"<ul>{itens}</ul>"
        )

    return f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{titulo_html}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f4f9;
                padding: 20px;
            }}
            .container {{
                max-width: 700px;
                margin: auto;
                padding: 24px;
                background: white;
                border-radius: 8px;
                overflow-wrap: anywhere;
            }}
            a {{ color: #007BFF; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>{titulo_html}</h2>
            <p>{mensagem_html}</p>
            {html_falhas}
            <p><a href="/">Voltar ao formulário</a></p>
        </div>
    </body>
    </html>
    """
```
