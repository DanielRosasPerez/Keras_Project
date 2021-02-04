import smtplib
from email.message import EmailMessage
import imghdr

def enviar_fotos(correo_origen, correo_destino, fotos_intrusos):
    """Está función envía las fotos de los intrusos al correo deseado"""
    
    assert type(correo_origen) == str and type(correo_destino) == str, "Ambos correos deben ser pasados como cadenas."

    msg = EmailMessage()

    msg["Subject"] = "¡INTRUSOS!"
    msg["From"] = correo_origen
    msg["To"] = correo_destino
    msg.set_content("A continuación, se muestran las fotos de los intrusos.")

    for foto in fotos_intrusos:

        with open(foto, 'rb') as img_a_adjuntar:
            imagen = img_a_adjuntar.read()
            tipo_archivo = imghdr.what(img_a_adjuntar.name)
            nombre_archivo = img_a_adjuntar.name

        msg.add_attachment(imagen, maintype="image", subtype=tipo_archivo, filename=nombre_archivo)

    with smtplib.SMTP_SSL(host="smtp.gmail.com", port=465) as smtp:

        smtp.login("TU_CORREO@gmail.com", "CONTRASEÑA")
        smtp.send_message(msg)
        
        print("¡Fotos enviadas con éxito al correo deseado!")