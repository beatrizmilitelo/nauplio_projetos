import tempfile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
  return """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Robô de Envio de E-mails</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 20px; }
            .container { max-width: 600px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin: auto; }
            h2 { color: #333; }
            label { font-weight: bold; display: block; margin-top: 10px; }
            input, textarea { width: 100%; padding: 8px; margin-top: 5px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
            button { background: #007BFF; color: white; border: none; padding: 10px 15px; margin-top: 20px; cursor: pointer; width: 100%; border-radius: 4px; font-size: 16px; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📧 Robô de Envio de E-mails</h2>
            <form action="/enviar" method="post" enctype="multipart/form-data">
                <label>E-mail do Gmail:</label>
                <input type="email" name="sender_email" required>

                <label>Senha de Aplicativo do Gmail:</label>
                <input type="password" name="sender_password" required>

                <label>Planilha de Destinatários (.xlsx):</label>
                <input type="file" name="excel_file" accept=".xlsx" required>

                <label>Imagem do Corpo do E-mail:</label>
                <input type="file" name="image_file" accept="image/*" required>

                <label>Assunto:</label>
                <input type="text" name="subject" value="Campanha Promocional" required>

                <label>HTML do E-mail:</label>
                <textarea name="body_html" rows="10" required><html>
<body>
<h1>Promoção Especial!</h1>
<p>Confira nossa promoção exclusiva.</p>
<img src="cid:image1">
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
  try:
    # Ler o excel
    contents_excel = await excel_file.read()
    df = pd.read_excel(contents_excel)

    if "Email" not in df.columns:
      return (
          "<h3>Erro: A planilha precisa possuir uma coluna chamada"
          ' Email.</h3><a href="/">Voltar</a>'
      )

    recipients = df["Email"].dropna().tolist()

    # Salvar imagem temporariamente
    contents_image = await image_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
      temp.write(contents_image)
      image_path = temp.name

    # Conectar ao SMTP
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_password)

    enviados = 0
    for recipient in recipients:
      try:
        msg = MIMEMultipart("related")
        msg["From"] = sender_email
        msg["To"] = recipient
        msg["Subject"] = subject

        body = MIMEMultipart("alternative")
        body.attach(MIMEText(body_html, "html", "utf-8"))
        msg.attach(body)

        with open(image_path, "rb") as f:
          img = MIMEImage(f.read())
          img.add_header("Content-ID", "<image1>")
          img.add_header(
              "Content-Disposition", "inline", filename="imagem.jpg"
          )
          msg.attach(img)

        server.sendmail(sender_email, recipient, msg.as_string())
        enviados += 1
      except Exception:
        pass

    server.quit()

    return f"""
        <h3>✅ {enviados} de {len(recipients)} e-mails enviados com sucesso!</h3>
        <br><a href="/">Voltar</a>
        """
  except Exception as e:
    return f"<h3>Erro ao processar: {str(e)}</h3><br><a href='/'>Voltar</a>"
