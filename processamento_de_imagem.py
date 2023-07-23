import cv2
import os

# Função para pré-processar uma imagem
def preprocess_image(input_path, output_path):
    # Carrega a imagem em RGB
    img = cv2.imread(input_path)
    
    # Converte a imagem para escala de cinza
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Alarga o histograma da imagem
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray_clahe = clahe.apply(cinza)

    # Equaliza o histograma da imagem
    img_eq = cv2.equalizeHist(gray_clahe)

    # Aplica o filtro de mediana
    img1 = cv2.medianBlur(img_eq, 1)
    
    # Aplica a binarização adaptativa
    #adaptive_thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Aplica a binarização adaptativa com parâmetros ajustados
    #block_size = 30  # tamanho da janela
    #C = 5  # valor de correção
    #adaptive_thresh = cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

    # Salva a imagem pré-processada
    cv2.imwrite(output_path, img1)

# Diretório das imagens de entrada
input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_cortadas_sem_defeito"
#input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_cortadas"
#input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\corte_novo"

# Diretório de saída para as imagens pré-processadas
#output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_processadas_sd"
#output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\Dataset\\NOT_OK"
output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\Dataset\\OK"
#output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_processadas_novo"

# Lista os arquivos do diretório de entrada
files = os.listdir(input_dir)

# Loop para pré-processar cada imagem
for file in files:
    # Caminho completo da imagem de entrada
    input_path = os.path.join(input_dir, file)
    
    # Caminho completo da imagem de saída
    output_path = os.path.join(output_dir, file)
    
    # Pré-processa a imagem
    preprocess_image(input_path, output_path)
