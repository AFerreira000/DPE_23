import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import h5py



# Lê a imagem
img = cv2.imread('C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_para_teste\\amolgadela_7.png', cv2.IMREAD_GRAYSCALE)

#### TESTE DE LER IMAGEM RECORTADA E PROCESSADA

#img = cv2.imread('C:\\Users\\andre\\Desktop\\Projeto DPE\\Dataset\\OK\\sem_defeito_20.png', cv2.IMREAD_GRAYSCALE)

#img = cv2.resize(img, (256, 256))
#img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

#### END TESTE DE LER IMAGEM RECORTADA E PROCESSADA

#RECORTE DA IMAGEM



# Aplica o filtro de mediana
img1 = cv2.medianBlur(img, 3)

cv2.imwrite('im_processada_1.png',img1)


# Cria o kernel
kernel5 = np.ones((20,20),np.float32)
kernel5[:,:5]    = 0.3
kernel5[:,5:10]  =-0.3
kernel5[:,10:15] = 0.3
kernel5[:,15:20] =-0.3

# Aplica o filtro
img2 = cv2.filter2D(img1,ddepth=0,kernel=kernel5)

cv2.imwrite('im_processada_2.png',img2)

# Aplica o threshold
ret, img3 = cv2.threshold(img2,130,255,cv2.THRESH_BINARY)

cv2.imwrite('im_processada_3.png',img3)


# Cria o kernel
kernel_dil = np.ones((10,10))

# Aplica a dilatação
img4 = cv2.dilate(img3,kernel_dil,iterations = 3)

cv2.imwrite('im_processada_4.png',img4)


# Binariza a imagem
thresh = cv2.threshold(img4, 127, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('im_processada_5.png',thresh)


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

# desenhar o retângulo na imagem
#cv2.rectangle(img, (x_centro, y_centro), (x_centro+largura, y_centro+altura), (255, 0, 0), 2)

# recorta a imagem
img_recortada = img[y_centro:y_centro + altura, x_centro:x_centro + largura]

cv2.imwrite('im_processada_6.png',img_recortada)


#PRE-PROCESSAMENTO DA IMAGEM
    
# Alarga o histograma da imagem
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(10,10))
gray_clahe = clahe.apply(img_recortada)

cv2.imwrite('im_processada_7.png',gray_clahe)

# Equaliza o histograma da imagem
img_eq = cv2.equalizeHist(gray_clahe)

#cv2.imwrite('im_processada_8.png',img_eq)



# REDIMENSIONAMENTO DA IMAGEM

#img_resized = cv2.resize(img_eq, (224, 224))
#img_resized = cv2.resize(img_eq, (256, 256))
img_resized = cv2.resize(img_eq, (500, 500))
#img_resized = cv2.resize(img, (500, 500))    # teste quando le imagem já processada

# Converter a imagem para o formato correto (RGB)
img_f = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)



cv2.imshow('img', img_f)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CARREGAR O MODELO

with h5py.File('C:\\Users\\andre\\Desktop\\Projeto DPE\\Modelo\\Modelo_sem_aumento_de_data4.h5', 'r') as f:
    model = tf.keras.models.load_model(f, compile=False)

# FAZER A PREDIÇÃO

# Normalizar a imagem
img_normalized = img_f / 255.0

# Redimensionar a imagem para ter uma dimensão extra para o tamanho do lote
input_img = np.expand_dims(img_normalized, axis=0)

print(input_img.shape)
print(input_img[0,:8,:8,2])

yhat = model.predict(input_img)

print(yhat)

if yhat[0][0] > 0.5: 
    print('Imagem OK')
else:
    print('Imagem NOT_OK')



