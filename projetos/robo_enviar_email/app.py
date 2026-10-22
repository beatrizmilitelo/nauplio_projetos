from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Robô de Envio de E-mails", page_icon="📧", layout="centered"
)

st.title("📧 Robô de Envio de E-mails com Anexos")
st.markdown("Preencha os dados abaixo para disparar sua campanha.")

with st.form("email_form"):
  sender_email = st.text_input("E-mail do Gmail:")
  sender_password = st.text_input(
      "Senha de Aplicativo do Gmail:", type="password"
  )

  excel_file = st.file_uploader(
      "Planilha de Destinatários (.xlsx):", type=["xlsx"]
  )
  attachment_files = st.file_uploader(
      "Arquivos em Anexo (PDF ou Imagens - selecione um ou mais):",
      type=["pdf", "png", "jpg", "jpeg"],
      accept_multiple_files=True,
  )

  subject = st.text_input("Assunto:", value="Campanha Promocional")

  default_html = """<html>
<body>
<h2>Promoção Especial!</h2>
<p>Confira os documentos e arquivos em anexo para mais detalhes.</p>
<p>Esperamos você!</p>
</body>
</html>"""
  body_html = st.text_area("Corpo do E-mail (HTML):", value=default_html, height=150)

  submit_button = st.form_submit_button("🚀 Enviar E-mails")

if submit_button:
  if not sender_email or not sender_password or not excel_file:
    st.error(
        "Por favor, preencha o e-mail, a senha e envie a planilha de"
        " destinatários."
    )
  else:
    try:
      with st.spinner("Processando e enviando e-mails..."):
        # Ler o excel
        df = pd.read_excel(excel_file)

        if "Email" not in df.columns:
          st.error("Erro: A planilha precisa possuir uma coluna chamada 'Email'.")
        else:
          recipients = df["Email"].dropna().tolist()

          # Ler os anexos
          attachments_data = []
          if attachment_files:
            for file in attachment_files:
              content = file.read()
              filename = file.name or "arquivo"
              attachments_data.append((content, filename))

          # Conectar ao SMTP
          server = smtplib.SMTP("smtp.gmail.com", 587)
          server.starttls()
          server.login(sender_email, sender_password)

          enviados = 0
          for recipient in recipients:
            try:
              msg = MIMEMultipart()
              msg["From"] = sender_email
              msg["To"] = recipient
              msg["Subject"] = subject

              # Corpo do e-mail em HTML
              msg.attach(MIMEText(body_html, "html", "utf-8"))

              # Anexar arquivos
              for file_bytes, filename in attachments_data:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(file_bytes)
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition", f"attachment; filename= {filename}"
                )
                msg.attach(part)

              server.sendmail(sender_email, recipient, msg.as_string())
              enviados += 1
            except Exception:
              pass

          server.quit()

          st.success(
              f"✅ {enviados} de {len(recipients)} e-mails enviados com"
              " sucesso!"
          )
    except Exception as e:
      st.error(f"Erro ao processar: {str(e)}")
