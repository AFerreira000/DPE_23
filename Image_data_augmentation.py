from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import os

datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=1.5,
        height_shift_range=1.5,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval=0)

input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_processadas_sd"
output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\Dataset1\\OK"

#input_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\imagens_processadas1"
#output_dir = "C:\\Users\\andre\\Desktop\\Projeto DPE\\Dataset1\\NOT_OK"

# Criando o diretório se ele não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Gerando as imagens aumentadas e salvando no diretório de saída
for root, dirs, files in os.walk(input_dir):
    for file in files:
        img_path = os.path.join(root, file)
        img = load_img(img_path, grayscale=True, target_size=(500, 500))
        x = img_to_array(img)  # this is a Numpy array with shape (150, 150, 1)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 150, 150, 1)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=output_dir, save_prefix='normal', save_format='png'):
            i += 1
            if i > 4:
                break  # otherwise the generator would loop indefinitely
