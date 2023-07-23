import cv2
import glob
import os
import numpy as np
import largestinteriorrectangle as lir

# cria a pasta para armazenar as imagens cortadas
#if not os.path.exists('imagens_cortadas'):
#    os.makedirs('imagens_cortadas')

# carrega as imagens da pasta
#imagens = glob.glob("C:\\Users\\andre\\Desktop\\Projeto DPE\\Defeitos2\\*")

# Diretório das imagens de entrada
#input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\sem_defeito1"
input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\Defeitos2"
#input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\Defeitos_novo"
#input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\teste_de_uma_img" #teste para uma imagem


# Diretório de saída para as imagens pré-processadas
#output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_cortadas_sem_defeito"
output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_cortadas"
#output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\corte_novo"
#output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE" #teste para uma imagem


# Função para recorta imagens
def recorte_imagem(input_path, output_path):

    # carrega a imagem
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) 

    #img = cv2.imread('C:\\Users\\andre\\Desktop\\Projeto DPE\\Defeitos2\\excesso_de_tinta_9.png', cv2.IMREAD_GRAYSCALE)

    
    # Aplica o filtro de mediana
    img1 = cv2.medianBlur(img, 3)

    cv2.imwrite('im_processada_1_1.png',img1)

    # Cria o kernel
    kernel5 = np.ones((20,20),np.float32)
    kernel5[:,:5]    = 0.3
    kernel5[:,5:10]  =-0.3
    kernel5[:,10:15] = 0.3
    kernel5[:,15:20] =-0.3

    # Aplica o filtro
    img2 = cv2.filter2D(img1,ddepth=0,kernel=kernel5)

    cv2.imwrite('im_processada_2_1.png',img2)

    # Aplica o threshold
    ret, img3 = cv2.threshold(img2,130,255,cv2.THRESH_BINARY)

    cv2.imwrite('im_processada_3_1.png',img3)

    # Cria o kernel
    kernel_dil = np.ones((10,10))

    # Aplica a dilatação
    img4 = cv2.dilate(img3,kernel_dil,iterations = 3)

    cv2.imwrite('im_processada_4_1.png',img4)

    # Binariza a imagem
    thresh = cv2.threshold(img4, 127, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite('im_processada_5_1.png',thresh)

    # encontrar os contornos na imagem
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontra o maior contorno
    max_contour = max(contours, key=cv2.contourArea)

    # preencher os contornos com a cor branca
    #cv2.drawContours(result, contours, -1, (255, 255, 255), cv2.FILLED)

    # salvar a imagem preenchida
    #cv2.imwrite('imagens_cortadas/teste6.jpg', result)


    # binariza a imagem
    #img_bin = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)

    #cv2.imshow('image', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Plota todos os contornos encontrados na imagem
    #cv2.drawContours(result, contours, -1, (0, 0, 255), 3)

    # Mostra a imagem com os contornos encontrados
    #cv2.imshow("All Contours", result)
    #cv2.waitKey(0)

    # salvar a imagem com o retângulo
    #cv2.imwrite('imagens_cortadas/teste7.jpg', result)

    # Encontra o maior contorno
    #max_contour = max(contours, key=cv2.contourArea)

    # Plota todos os contornos encontrados na imagem
    #cv2.drawContours(img, max_contour, -1, (0, 0, 255), 3)

    # Mostra a imagem com o maior contorno encontrado
    #cv2.imshow("bigest Contours", img)
    #cv2.waitKey(0)

    # preencher os contornos com a cor branca
    #cv2.drawContours(result1, max_contour, -1, (255, 255, 255), cv2.FILLED)

    # exibir a imagem com o maior retângulo possível
    #cv2.imshow('Maior Retângulo Possível', result1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Converte o contorno para um array
    points = np.squeeze(max_contour).tolist()
    points = np.array(points)
    


    # verificar se os dados de entrada são int32
    if points.dtype != np.int32:
        # converter os dados de entrada para int32, se necessário
        points = points.astype(np.int32)

    # Encontra o retângulo que envolve o contorno
    x, y, w, h = cv2.boundingRect(points)

    # Calcular coordenadas do maior retângulo dentro do contorno
    largura = w - 50 # ou qualquer outro valor que você queira subtrair da largura
    altura = h - 100 # ou qualquer outro valor que você queira subtrair da altura
    x_centro = x + w//2 - largura//2
    y_centro = y + h//2 - altura//2

    # desenhar o retângulo na imagem
    #cv2.rectangle(result1, (x_centro, y_centro), (x_centro+largura, y_centro+altura), (255, 0, 0), 2)

    # salvar a imagem com o retângulo
    #cv2.imwrite('imagens_cortadas/teste7.jpg', result1)

    # exibir a imagem com o maior retângulo possível
    #cv2.imshow('Maior Retângulo Possível', result1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # recorta a imagem
    crop_img = img[y_centro:y_centro + altura, x_centro:x_centro + largura]

    cv2.imwrite('im_processada_6_1.png',crop_img)

    # exibe a imagem recortada
    #cv2.imshow('Imagem Recortada', crop_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #imagem_path = 'C:\\Users\\andre\\Desktop\\Projeto DPE\\Defeitos2\\*'

    # salvar a imagem cortada
    cv2.imwrite(output_path, crop_img)


# Lista os arquivos do diretório de entrada
files = os.listdir(input_dir)

# Loop para recortar cada imagem
for file in files:
    # Caminho completo da imagem de entrada
    input_path = os.path.join(input_dir, file)
    
    
    # Caminho completo da imagem de saída
    output_path = os.path.join(output_dir, file)
    
    # Pré-processa a imagem
    recorte_imagem(input_path, output_path)
