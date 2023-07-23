import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input 
import h5py
from tensorflow.keras.models import load_model, model_from_json, model_from_yaml, Sequential, Model, save_model
import minhas_funções


# Carregar o modelo
with h5py.File('C:\\Users\\andre\\Desktop\\Projeto DPE\\Modelo\\Funciona_Modelo_sem_aumento_de_data4.h5', 'r') as f:
    model = tf.keras.models.load_model(f, compile=False)

# Carregar o modelo
with h5py.File('C:\\Users\\andre\\Desktop\\Projeto DPE\\Modelo\\Modelo_tipo_de_defeito2.h5', 'r') as f:
    model_1 = tf.keras.models.load_model(f, compile=False)

# Criar a janela principal
window = tk.Tk()
window.title("Interface")  # Definir o título da janela

accuracy_label = tk.Label(window, text="")
accuracy_label.pack()


def load_image():
    # Abrir a janela de diálogo para selecionar o arquivo da imagem
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

    # Verificar se um arquivo foi selecionado
    if file_path:
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        img_recortada = minhas_funções.recorte_de_uma_imagem(img)

        img_processada = minhas_funções.processamento_da_imagem(img_recortada)

        # REDIMENSIONAMENTO DA IMAGEM

        #img_resized = cv2.resize(img_eq, (224, 224))
        img_resized = cv2.resize(img_processada, (500, 500))
        #img_resized = cv2.resize(img_processada, (600, 400))

        # Converter a imagem para o formato correto (RGB)
        img_f = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # FAZER A PREDIÇÃO

        # Normalizar a imagem
        img_normalized = img_f / 255.0

        # Redimensionar a imagem para ter uma dimensão extra para o tamanho do lote
        input_img = np.expand_dims(img_normalized, axis=0)
        

        yhat = model.predict(input_img)

        print(yhat)

        # Limpar o texto da prediction2
        prediction_label2.config(text="")

        # Verificar a previsão
        if yhat[0][0] > 0.5:
            prediction = 'Imagem OK'
            confidence = yhat[0][0] * 100
        
        else:
            prediction = 'Imagem NOT_OK'
            confidence = (1 - yhat[0][0])  * 100
            yhat1 = model_1.predict(input_img)
            m = yhat1[0][0]
            idx = 0
            for i in range(1,4):
                if yhat1[0][i]> m :
                    m = yhat1[0][i]
                    idx = i
            print(yhat1)
            print(idx, m)
            if idx == 0 :
                prediction1 = 'Amolgadela'
            elif idx == 1:
                prediction1 = 'Excesso de tinta'
            elif idx == 2:
                prediction1 = 'Risco'
            else:
                prediction1 = 'Sujidade'

            prediction_label2.config(text=prediction1)


        # Exibir a previsão na interface gráfica
        prediction_label.config(text=prediction)
        
        

        # Atualizar o texto da etiqueta accuracy_label
        accuracy_label.config(text=f"Confiança: {confidence:.2f}%")
        

        # Exibir a imagem original
        img_original = Image.fromarray(img)
        img_original = img_original.resize((600, 400))  # Ajuste o tamanho conforme necessário
        img_original_tk = ImageTk.PhotoImage(img_original)
        img_original_label.config(image=img_original_tk)
        img_original_label.image = img_original_tk

        # Exibir a imagem final
        img_final = Image.fromarray(img_f)
        img_final = img_final.resize((600, 400))  # Ajuste o tamanho conforme necessário
        img_final_tk = ImageTk.PhotoImage(img_final)
        img_final_label.config(image=img_final_tk)
        img_final_label.image = img_final_tk



# Criar um botão para carregar a imagem
load_button = tk.Button(window, text="Carregar Imagem", command=load_image)
load_button.pack()

# Criar uma etiqueta para exibir a previsão
prediction_label = tk.Label(window, text="")
prediction_label.pack()

# Criar uma etiqueta para exibir a previsão 2
prediction_label2 = tk.Label(window, text="")
prediction_label2.pack()


# Criar uma etiqueta para exibir a imagem original
img_original_label = tk.Label(window)
img_original_label.pack(side=tk.LEFT)

# Criar uma etiqueta para exibir a imagem final
img_final_label = tk.Label(window)
img_final_label.pack(side=tk.RIGHT)

# Executar o loop principal da janela
window.mainloop()