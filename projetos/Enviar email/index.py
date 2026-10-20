import smtplib
import pandas as pd
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_bulk_email_with_embedded_image(sender_email, sender_password, df, subject, body_html, image_file, email_column):
    recipients = df[email_column].tolist()
    
    try:
        # Configurar o servidor SMTP do Outlook
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
    except Exception as e:
        return False, f"Erro ao conectar ao servidor SMTP: {e}"

    progress_bar = st.progress(0)
    total = len(recipients)

    for i, recipient in enumerate(recipients):
        try:
            msg = MIMEMultipart('related')
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject

            body = MIMEMultipart('alternative')
            body.attach(MIMEText(body_html, 'html'))
            msg.attach(body)

            # Ler os bytes da imagem enviada pelo Streamlit
            image_bytes = image_file.getvalue()
            img = MIMEImage(image_bytes)
            img.add_header('Content-ID', '<image1>')
            img.add_header('Content-Disposition', 'inline', filename=image_file.name)
            msg.attach(img)

            server.sendmail(sender_email, recipient, msg.as_string())
            
            # Atualiza a barra de progresso
            progress_bar.progress((i + 1) / total)
        except Exception as e:
            st.warning(f"Erro ao enviar para {recipient}: {e}")

    server.quit()
    return True, "E-mails enviados com sucesso!"

# Interface do Streamlit
st.set_page_config(page_title="Disparador de E-mails", page_icon="📧", layout="centered")

st.title("📧 Disparador de E-mails em Massa")
st.markdown("Envie campanhas personalizadas com imagens embutidas usando sua conta do Outlook/Office365.")

with st.form("email_form"):
    st.subheader("1. Credenciais e Configurações")
    col1, col2 = st.columns(2)
    with col1:
        sender_email = st.text_input("E-mail do Remetente", placeholder="seu-email@outlook.com")
    with col2:
        sender_password = st.text_input("Senha ou App Password", type="password")
        
    subject = st.text_input("Assunto do E-mail", value="Campanha Promocional")
    
    st.subheader("2. Arquivos (Planilha e Imagem)")
    uploaded_excel = st.file_uploader("Envie a planilha (Excel .xlsx)", type=["xlsx"])
    
    email_column = "Email"
    if uploaded_excel:
        df_preview = pd.read_excel(uploaded_excel)
        columns = df_preview.columns.tolist()
        email_column = st.selectbox("Selecione a coluna que contém os e-mails:", columns, index=columns.index("Email") if "Email" in columns else 0)
    
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
    
    submitted = st.form_submit_button("🚀 Enviar E-mails")

if submitted:
    if not sender_email or not sender_password:
        st.error("Por favor, preencha o e-mail e a senha do remetente.")
    elif not uploaded_excel:
        st.error("Por favor, envie a planilha Excel.")
    elif not uploaded_image:
        st.error("Por favor, envie a imagem.")
    else:
        df = pd.read_excel(uploaded_excel)
        with st.spinner("Enviando e-mails, por favor aguarde..."):
            success, message = send_bulk_email_with_embedded_image(
                sender_email, sender_password, df, subject, body_html, uploaded_image, email_column
            )
            if success:
                st.success(message)
            else:
                st.error(message)
