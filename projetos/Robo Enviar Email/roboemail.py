import streamlit as st
import pandas as pd
import smtplib
import tempfile

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


# ===============================
# CONFIGURAÇÃO DA PÁGINA
# ===============================

st.set_page_config(
    page_title="Robô de Envio de E-mails",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Robô de Envio de E-mails")
st.write("Envie campanhas de e-mail utilizando uma conta Gmail.")

st.divider()


# ===============================
# DADOS DO REMETENTE
# ===============================

st.header("🔐 Dados do Remetente")

sender_email = st.text_input(
    "E-mail do Gmail"
)

sender_password = st.text_input(
    "Senha de Aplicativo do Gmail",
    type="password"
)

st.info(
    """
A senha utilizada **não é a senha da conta Google**.

É necessário criar uma **Senha de Aplicativo** em:

https://myaccount.google.com/apppasswords
"""
)

st.divider()


# ===============================
# PLANILHA
# ===============================

st.header("📄 Planilha de Destinatários")

st.markdown("""
### Formato esperado

A planilha deve ser um arquivo **.xlsx** contendo obrigatoriamente a coluna:

| Email |
|-------|
| cliente@email.com |
| contato@empresa.com |

Cada linha representa um destinatário.

Outras colunas podem existir, porém atualmente apenas **Email** será utilizada.
""")

excel_file = st.file_uploader(
    "Selecione a planilha",
    type=["xlsx"]
)

st.divider()


# ===============================
# IMAGEM
# ===============================

st.header("🖼 Imagem")

st.write("Selecione a imagem que será exibida no corpo do e-mail.")

image_file = st.file_uploader(
    "Imagem",
    type=["png", "jpg", "jpeg"]
)

st.divider()


# ===============================
# E-MAIL
# ===============================

st.header("✉ Conteúdo")

subject = st.text_input(
    "Assunto",
    value="Campanha Promocional"
)

body_html = st.text_area(
    "HTML",
    height=350,
    value="""
<html>
<body>

<h1>Promoção Especial!</h1>

<p>Confira nossa promoção exclusiva.</p>

<img src="cid:image1">

<p>Esperamos você!</p>

</body>
</html>
"""
)

st.divider()


# ===============================
# BOTÃO
# ===============================

if st.button("🚀 Enviar E-mails", use_container_width=True):

    if sender_email == "":
        st.error("Informe o e-mail do Gmail.")
        st.stop()

    if sender_password == "":
        st.error("Informe a senha de aplicativo.")
        st.stop()

    if excel_file is None:
        st.error("Envie uma planilha.")
        st.stop()

    if image_file is None:
        st.error("Envie uma imagem.")
        st.stop()

    try:

        df = pd.read_excel(excel_file)

        if "Email" not in df.columns:
            st.error("A planilha precisa possuir uma coluna chamada Email.")
            st.stop()

        recipients = df["Email"].dropna().tolist()

        with tempfile.NamedTemporaryFile(delete=False) as temp:

            temp.write(image_file.read())
            image_path = temp.name

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        server.login(
            sender_email,
            sender_password
        )

        progresso = st.progress(0)

        status = st.empty()

        enviados = 0

        total = len(recipients)

        for i, recipient in enumerate(recipients):

            try:

                msg = MIMEMultipart("related")

                msg["From"] = sender_email
                msg["To"] = recipient
                msg["Subject"] = subject

                body = MIMEMultipart("alternative")

                body.attach(
                    MIMEText(
                        body_html,
                        "html",
                        "utf-8"
                    )
                )

                msg.attach(body)

                with open(image_path, "rb") as f:

                    img = MIMEImage(f.read())

                    img.add_header(
                        "Content-ID",
                        "<image1>"
                    )

                    img.add_header(
                        "Content-Disposition",
                        "inline",
                        filename="imagem.jpg"
                    )

                    msg.attach(img)

                server.sendmail(
                    sender_email,
                    recipient,
                    msg.as_string()
                )

                enviados += 1

            except Exception as erro:

                st.warning(f"{recipient}: {erro}")

            progresso.progress((i + 1) / total)

            status.write(
                f"Enviando {i+1} de {total}"
            )

        server.quit()

        st.success(
            f"✅ {enviados} e-mails enviados com sucesso!"
        )

    except Exception as erro:

        st.error(str(erro))
