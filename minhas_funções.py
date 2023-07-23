import cv2
import numpy as np


def recorte_de_uma_imagem(input_image):
    
    
    # Aplica o filtro de mediana
    img1 = cv2.medianBlur(input_image, 3)

    # Cria o kernel
    kernel5 = np.ones((20,20),np.float32)
    kernel5[:,:5]    = 0.3
    kernel5[:,5:10]  =-0.3
    kernel5[:,10:15] = 0.3
    kernel5[:,15:20] =-0.3

    # Aplica o filtro
    img2 = cv2.filter2D(img1,ddepth=0,kernel=kernel5)

    # Aplica o threshold
    ret, img3 = cv2.threshold(img2,130,255,cv2.THRESH_BINARY)

    # Cria o kernel
    kernel_dil = np.ones((10,10))

    # Aplica a dilatação
    img4 = cv2.dilate(img3,kernel_dil,iterations = 3)

    # Binariza a imagem
    thresh = cv2.threshold(img4, 127, 255, cv2.THRESH_BINARY)[1]

    # Encontra os contornos na imagem binarizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Encontra o maior contorno
    max_contour = max(contours, key=cv2.contourArea)

    # Converte o contorno para um array
    points = np.squeeze(max_contour).tolist()
    points = np.array(points)

    # verificar se os dados de entrada são int32
    if points.dtype != np.int32:
        # converter os dados de entrada para int32, se necessário
        points = points.astype(np.int32)

    # Encontra o retângulo que envolve o contorno
    x, y, w, h = cv2.boundingRect(points)

    print(x,y,w,h)

    # Calcular coordenadas do retângulo dentro do contorno
    largura = w - 50 # ou qualquer outro valor que você queira subtrair da largura
    altura = h - 100 # ou qualquer outro valor que você queira subtrair da altura
    x_centro = x + w//2 - largura//2
    y_centro = y + h//2 - altura//2

    # recorta a imagem
    img_recortada = input_image[y_centro:y_centro + altura, x_centro:x_centro + largura]

    return img_recortada



def processamento_da_imagem (input_image):
    
    # Alarga o histograma da imagem
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray_clahe = clahe.apply(input_image)

    # Equaliza o histograma da imagem
    img_eq = cv2.equalizeHist(gray_clahe)

    # Aplica o filtro de mediana
    img1 = cv2.medianBlur(img_eq, 1)

    return img1