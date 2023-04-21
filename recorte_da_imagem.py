import cv2
import glob
import os
import numpy as np

# cria a pasta para armazenar as imagens cortadas
if not os.path.exists('imagens_cortadas'):
    os.makedirs('imagens_cortadas')

# carrega as imagens da pasta
imagens = glob.glob("C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens\\*")

# define o tamanho desejado para as imagens
tamanho_imagem = (500, 500)

# redimensiona as imagens
for imagem in imagens:
    img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, tamanho_imagem)
    cv2.imwrite(imagem, img)

# carrega o padrão
padrao = cv2.imread("C:\\Users\\andre\\Desktop\\Projeto DPE\\padrao_corte.png", 0)

print(padrao.shape)

# redimensiona o padrão
padrao = cv2.resize(padrao, tamanho_imagem)


print(padrao.shape)

# cria o detector de características (nesse exemplo, utilizando ORB)
detector_caracteristicas = cv2.ORB_create(nfeatures=500)

for imagem in imagens:
    # carrega a imagem
    img = cv2.imread(imagem, cv2.IMREAD_COLOR).astype(np.uint8)

    # detecta as características da imagem e do padrão
    kp1, des1 = detector_caracteristicas.detectAndCompute(img, None)
    kp2, des2 = detector_caracteristicas.detectAndCompute(padrao, None)

    # Convert des1 and des2 to np.float32
    des1 = np.asarray(des1)
    des2 = np.asarray(des2)

    print(des1)
    print(des2)

    print(des1.shape)
    # realiza a correspondência de características
    correspondencias = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = correspondencias.match(des1, des2)

    # filtra as correspondências para obter as melhores correspondências
    matches = sorted(matches, key=lambda x: x.distance)
    melhores_matches = matches[:10]

    # obtém as coordenadas do padrão na imagem
    pontos_imagem = []
    pontos_padrao = []
    for match in melhores_matches:
        pontos_imagem.append(kp1[match.queryIdx].pt)
        pontos_padrao.append(kp2[match.trainIdx].pt)

    pontos_imagem = np.float32(pontos_imagem)
    pontos_padrao = np.float32(pontos_padrao)

    # realiza a transformação perspectiva para cortar a imagem
    M, _ = cv2.findHomography(pontos_padrao, pontos_imagem, cv2.RANSAC)
    altura, largura = padrao.shape[:2]
    img_cortada = cv2.warpPerspective(img, M, (largura, altura))

    # salva a imagem cortada na pasta 'imagens_cortadas'
    nome_imagem = os.path.splitext(os.path.basename(imagem))[0]
    cv2.imwrite('imagens_cortadas/{}_cortada.jpg'.format(nome_imagem), img_cortada)

