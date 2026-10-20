import smtplib
import pandas as pd
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_bulk_email_with_embedded_image(sender_email, sender_password, df, subject, body_html, image_file):
    recipients = df.iloc[:, 0].dropna().astype(str).tolist()
    
    # Identifica o servidor SMTP correto com base no domínio do e-mail
    sender_email_lower = sender_email.strip().lower()
    if "gmail.com" in sender_email_lower:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
    elif "outlook" in sender_email_lower or "hotmail" in sender_email_lower or "live" in sender_email_lower or "office365" in sender_email_lower:
        smtp_server = "smtp.office365.com"
        smtp_port = 587
    else:
        return False, "Domínio de e-mail não reconhecido automaticamente. Use um e-mail do Gmail ou Outlook/Hotmail."

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
    except Exception as e:
        return False, f"Erro ao conectar ou autenticar no servidor SMTP: {e}"

    progress_bar = st.progress(0)
    total = len(recipients)

    for i, recipient in enumerate(recipients):
        if "@" not in recipient:
            continue
            
        try:
            msg = MIMEMultipart('related')
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject

            body = MIMEMultipart('alternative')
            body.attach(MIMEText(body_html, 'html'))
            msg.attach(body)

            image_bytes = image_file.getvalue()
            img = MIMEImage(image_bytes)
            img.add_header('Content-ID', '<image1>')
            img.add_header('Content-Disposition', 'inline', filename=image_file.name)
            msg.attach(img)

            server.sendmail(sender_email, recipient, msg.as_string())
            progress_bar.progress((i + 1) / total)
        except Exception as e:
            st.warning(f"Erro ao enviar para {recipient}: {e}")

    server.quit()
    return True, "E-mails enviados com sucesso!"

st.set_page_config(page_title="Disparador de E-mails", page_icon="📧", layout="centered")

st.title("📧 Disparador de E-mails em Massa")
st.markdown("Envie campanhas personalizadas com imagens embutidas usando contas do **Outlook** ou **Gmail**.")

st.subheader("1. Credenciais e Configurações")
col1, col2 = st.columns(2)
with col1:
    sender_email = st.text_input("E-mail do Remetente", placeholder="seu-email@gmail.com ou @outlook.com")
with col2:
    sender_password = st.text_input("Senha ou Senha de App", type="password")
    
subject = st.text_input("Assunto do E-mail", value="Campanha Promocional")

st.subheader("2. Arquivos (Planilha e Imagem)")
uploaded_excel = st.file_uploader("Envie a planilha (Excel .xlsx)", type=["xlsx"])

st.info("""
📌 **Como montar sua planilha Excel (.xlsx):**
- A planilha deve conter **apenas uma coluna** com os e-mails.
- **Não coloque cabeçalho** (comece a colocar os e-mails direto na primeira linha da primeira coluna).
- Exemplo:
  
  | Coluna A |
  | :--- |
  | cliente1@exemplo.com |
  | cliente2@exemplo.com |
""")

uploaded_image = st.file_uploader("Envie a imagem para o corpo do e-mail", type=["png", "jpg", "jpeg"])

st.subheader("3. Corpo do E-mail (HTML)")
default_html = """<html>
  <body>
    <h2>Promoção Especial!</h2>
    <p>Confira nossa promoção exclusiva para você.</p>
    <img src="cid:image1" alt="Imagem Promocional" style="max-width:100%;">
    <p>Não perca!</p>
  </body>
</html>"""
body_html = st.text_area("Código HTML do E-mail", value=default_html, height=150)

if st.button("🚀 Enviar E-mails", type="primary"):
    if not sender_email or not sender_password:
        st.error("Por favor, preencha o e-mail e a senha do remetente.")
    elif not uploaded_excel:
        st.error("Por favor, envie a planilha Excel.")
    elif not uploaded_image:
        st.error("Por favor, envie a imagem.")
    else:
        df = pd.read_excel(uploaded_excel, header=None)
        with st.spinner("Enviando e-mails, por favor aguarde..."):
            success, message = send_bulk_email_with_embedded_image(
                sender_email, sender_password, df, subject, body_html, uploaded_image
            )
            if success:
                st.success(message)
            else:
                st.error(message)
