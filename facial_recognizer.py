import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, io
import skimage.io, skimage.feature
from PIL import Image

# Filtros de convolución:

# Creamos el banco de filtros:
Filtros = np.zeros(shape=(2,3,3))

# Este primer filtro detectará los ejes horizontales:
Filtros[0,:,:] = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

# Este segundo filtro detectará los ejes verticales:
Filtros[1,:,:] = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

# A continuación, reacomodamos el nuestra matriz de filtros, para poder aprovechar la rápidez del producto punto en numpy:
Filtros = Filtros.reshape(2,-1).T

#########################################################################################################

from Modulos import *
from Preprocesado_imagenes import *
from keras.models import load_model

anchura, altura = 150,150

net = cv2.dnn.readNetFromDarknet('./cfg/yolov3-tiny.cfg', './model-weights/face-yolov3-tiny_41000.weights') # YOLOv3 Ligera.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture(0)

# Cargamos nuestro modelo para el reconcimiento de rostros:
cnn_reconocimiento_rostro = load_model("Clasificador_Bordes_1000.h5")

# Declaramos un diccionario para saber que persona se encuentra en cámara:
Personas = {0: "Dany", 1:"Diana", 2:"Mama", 3:"Papa", 4:"Tripi"}

def preprocesado_patches(patch, diccionario_personas, cnn_reconocimiento_rostro):
    """Esta función preprocesa el patch previo a ser ingreado a la red neuronal para su predicción."""
    
    #patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)/255.0
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = convolucion(patch, Filtros)
    patch.resize((1,anchura,altura,1))
    
    Vector_Prediccion = cnn_reconocimiento_rostro.predict(patch)[0]
    Clase = np.argmax(Vector_Prediccion)
    Persona_Inferida = diccionario_personas[Clase]
    
    return Persona_Inferida, Vector_Prediccion

def detector_rostro(cnn_reconocimiento_rostro):
    
    from collections import Counter
    
    #salida = cv2.VideoWriter('Video_Bordes_30.mp4',cv2.VideoWriter_fourcc(*'XVID'),10.0,(640,480))
    
    wind_name = 'Detectar Rostros Usando Tiny-YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    
    lista_ID = list()
    flag = 0
    intrusos = list()
    rostro_sospechoso = list()

    while True:

        success, frame = cap.read()

        if success:
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(get_outputs_names(net))

            # Remove the bounding boxes with low confidence
            faces, patches, coordenadas = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            
            if faces: # Para evitar errores cuando nuestro rostro sale de cuadro.
                for num, (rostro, coords) in enumerate(zip(patches, coordenadas), start=1):
                
                    # Preprocesamos los rostros y realizamos la inferencia:
                    Persona_Inferida, Vector_Prediccion = preprocesado_patches(rostro, Personas, cnn_reconocimiento_rostro)
                    
                    x_1 = coords[0][0]
                    y_1 = coords[0][1]
                    x_2 = coords[1][0]
                    y_2 = coords[1][1]
                    
                    prob = np.around(Vector_Prediccion.max(), decimals=4)
                    cv2.putText(frame, str(prob)+'%', (x_2-60,y_2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                    if prob >= 0.96:
                        Desconocido = 0
                        cv2.putText(frame, str(num) + ' ' + Persona_Inferida, (x_1,y_1+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        
                        
                    else:
                        Desconocido = 1
                        cv2.putText(frame, str(num) + ' ' + "Intruso", (x_1,y_1+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    
                    if Desconocido == 1: # Es decir, el rostro localizado es desconocido.
                        rostro_sospechoso.append((num,rostro))
            
                # initialize the set of information we'll display it on the frame:
                info = [
                    ('Numero de Rostros Detectado: ', '{}'.format(len(faces)))
                ]

                for (i, (txt, val)) in enumerate(info):
                    text = '{}: {}'.format(txt, val)
                    cv2.putText(frame, text, (10, (i * 20) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            
            #salida.write(frame)
            
            # Detectamos posibles sospechosos cada 15 frames:
            if flag >= 15:
                if rostro_sospechoso: # En caso de que se detecten caras desconocidas dentro de los 15 frames.
                    for ID,aparicion in Counter([n for n,r in rostro_sospechoso]).most_common():
                        if aparicion >= 8: # Si en 15 frames la persona se consideró más de 8 veces como Desconocido, entonces pasa a ser un sospechoso.
                            print(f"¡El rostro número {ID} no se reconoce!")
                            for num, (id_,sospechoso) in enumerate(rostro_sospechoso):
                                if id_ == ID and num < 3: # Para adjuntar únicamente 3 fotos por sospechoso:
                                    intrusos.append(cv2.cvtColor(sospechoso, cv2.COLOR_RGB2BGR))
                                    
                    flag = 0
                    rostro_sospechoso = list()
                    
            if len(intrusos) >= 20:
                
                for i in range(len(intrusos)//2):
                    intrusos.pop(i)
            
            # Showing the video with the faces in it:
            
            cv2.imshow(wind_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                if bool(intrusos) == True:
                    
                    cap.release()
                    cv2.destroyAllWindows()

                    print('==> Terminado!')
                    print('***********************************************************')
                                
                    return intrusos
                
                print('[i] ==> Interrumpido por el usuario!')
                break
                
            flag += 1
            
                
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> Terminado!')
    print('***********************************************************')
    
    return False
    
#########################################################################################################

posibles_sospechosos = detector_rostro(cnn_reconocimiento_rostro) # Llamamos a la función.

#########################################################################################################
# Mandamos las fotos de los sospechosos por correo:

if posibles_sospechosos: # En caso de haya algún sospechoso.

    nuevos_sospechosos = np.asarray(posibles_sospechosos).shape[0] # Obtenemos la cantidad total de rostros sospechosos.

    dir_sospechosos = os.listdir("Rostro_Sospechosos/")

    for i,rostro_sospechoso in enumerate(posibles_sospechosos, start=len(dir_sospechosos)):
        skimage.io.imsave(fname=f"Rostro_Sospechosos/Sospechoso{i+1}.jpg",arr=rostro_sospechoso)

    import glob

    sospechosos_en_dir = glob.glob("Rostro_Sospechosos/*.jpg") # Obtenemos todos aquellos archivos dentro de la carpeta cuya terminación sea ".jpg".
    sospechosos_en_dir.sort(key=os.path.getmtime) # Ordenamos los archivos obtenidos por su fecha de creación, del menos reciente al más reciente.

    from Email import enviar_fotos

    fotos = [sospechoso for sospechoso in sospechosos_en_dir[-nuevos_sospechosos:]]

    enviar_fotos("daniel020197ss@gmail.com","daniel020197ss@gmail.com",fotos) # Enviamos las fotos al correo.

else:

    print("¡No se encontraron sospechosos!")
