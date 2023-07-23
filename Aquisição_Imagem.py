import os
from vimba import *
import numpy as np
import cv2
import time

# Defina o número de imagens que você deseja coletar
num_imagens = 100

count = 0


# Inicie o gerenciador de câmera Vimba
with Vimba.get_instance() as vimba:
    # Lista de câmeras disponíveis
    camera_ids = vimba.get_all_cameras()
    print(f"Found {len(camera_ids)} camera(s)")
    
    # Imprimir lista de câmeras
    print('Câmeras disponíveis:')
    for camera_id in camera_ids:
        print(camera_id)
     
    # Abrir a câmera
    camera_id = camera_ids[0] 
    #camera_id.UserSetLoad.run()
    print('AA') 

    with camera_id as camera:
        print('AB') 
        # Parametros da câmera
        #camera.PixelFormat = 'Mono8'
        #camera.set_binning_horizontal(8)
        camera.load_settings('settings8.xml', PersistType.All)
        #camera.UserSetLoad.run()
        #camera.BinningVertical = 8
            
        # Inicia a aquisição de imagens
        while count < num_imagens:
            print('AC') 
            frame = camera.get_frame()
            
            image_data = frame.as_numpy_ndarray()
            image_data = image_data.astype(np.uint8)
            
            # Converte a imagem para o formato do OpenCV 
            image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

            # Redimensiona a imagem para um tamanho menor
            #scale_percent = 19  # Defina a porcentagem em relação à original
            #width = int(image.shape[1] * scale_percent / 100)
            #height = int(image.shape[0] * scale_percent / 100)
            #dim = (width, height)
            #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # Mostre a imagem na tela
            cv2.imshow('Image', image)

            # Aguarde até que uma tecla seja pressionada
            key = cv2.waitKey(0)

            # Se a tecla pressionada for 's', salve a imagem na pasta 'imagens' com um nome de arquivo único
            if key == ord('s'):
                arquivo = os.path.join('C:\\Users\\andre\\Desktop\\Projeto DPE\\Defeitos_novo', 'excesso_de_tinta1_{}.png'.format(count))
                cv2.imwrite(arquivo, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print('Imagem {} salva!'.format(count))
                count = count + 1
            # Se a tecla pressionada for 'q', saia do loop
            elif key == ord('q'):
                break


        # Para a aquisição de imagens
        cv2.destroyAllWindows()

# arquivo = os.path.join('C:\\Users\\andre\\Desktop\\Projeto DPE\\Defeitos2', 'imagem_{}_{}.png'.format(count, time.time()))