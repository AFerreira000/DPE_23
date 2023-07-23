import cv2
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.models import load_model

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

data = tf.keras.utils.image_dataset_from_directory('C:\\Users\\andre\\Desktop\\Projeto DPE\\Dataset_defeito', batch_size=8, image_size=(500, 500)) #batch_size=32, image_size=(256, 256), shuffle=True, seed=123, validation_split=0.2, subset="training", interpolation="bilinear", follow_links=False
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

print(batch[0].shape)

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

print(batch[1]) # No batch[0] temos as imagens e no batch[1] temos as classes
plt.show()

# classe 0 = amolgadela
# classe 1 = excesso de tinta
# classe 2 = risco
# classe 3 = sujidade


# PREPROCESSAMENTO DE IMAGENS 
data = data.map(lambda x,y: (x/255, y)) #Scale the images to [0,1]
data.as_numpy_iterator().next()

# DIVISÃO DE DADOS PARA TREINO E TESTE
print(len(data))
train_size = int(len(data)*.6)
val_size = int(len(data)*.2)  # pode ser necessário ajustar o tamanho par que a soma dos 3 seja = len(data)
test_size = int(len(data)*.2)



train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# CRIAÇÃO DO MODELO
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(500,500,3)))
#model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D())

#model.add(Conv2D(256, (3,3), 1, activation='relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))


model.compile('Adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']) #Existem diferentes otimizadores e neste caso estou a usar o adam 

print(model.summary())

# TREINO DO MODELO
logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

# PLOT DO GRÁFICO DE TREINO

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig1 = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig1.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


'''
# AVALIAÇÃO DO MODELO
pre = Precision()
re = Recall()
acc = SparseCategoricalAccuracy()


for batch in test.as_numpy_iterator(): 
    X, y_true = batch
    #print(X[0,:4,:4,0])
    #print(X.shape, y.shape)
    #y_true = np.argmax(y_true, axis=1)
    yhat = model.predict(X)
    yhat = np.argmax(yhat, axis=1)
    yhat = tf.one_hot(yhat, depth=4)
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

'''

pre = 0
re = 0
acc = 0
total_samples = 0

for batch in test.as_numpy_iterator():
    X, y_true = batch
    yhat = model.predict(X)
    yhat = np.argmax(yhat, axis=1)
    
    pre += np.sum((yhat == y_true) & (yhat != 0)) / np.sum(yhat != 0)
    re += np.sum((yhat == y_true) & (y_true != 0)) / np.sum(y_true != 0)
    acc += np.sum(yhat == y_true)
    
    total_samples += len(y_true)

pre /= total_samples
re /= total_samples
acc /= total_samples

print(f'Accuracy: {acc}')





model.save(os.path.join('Modelo','Modelo_tipo_de_defeito2.h5'))



