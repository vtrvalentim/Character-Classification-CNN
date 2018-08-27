#Module 1 creates inputs for all 26 characters/classes
#Module 2 creates inputs for only 5 characters/classes and should be used for fast prototyping

from module2 import CreateInput
#from module1 import CreateInput
import tensorflow as tf
import keras as K
import pydot
import graphviz


if __name__ == '__main__':
    
    train_data, train_label, test_data, test_label = CreateInput()
    
    model = K.Sequential()
    
    model.add(K.layers.Conv2D(4,(31,31),strides=(1,1),padding='valid',activation='relu',input_shape=(128,128,3)))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    
    model.add(K.layers.Conv2D(8,(16,16),strides=(1,1),padding='valid',activation='relu'))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    
    model.add(K.layers.Conv2D(16,(6,6),strides=(1,1),padding='valid',activation='relu'))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    
    
    model.add(K.layers.Dense(128,activation='relu'))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dropout(0.1))
    model.add(K.layers.Dense(64,activation='relu'))
    model.add(K.layers.Dropout(0.1))
    model.add(K.layers.Dense(32,activation='relu'))
    model.add(K.layers.Dense(5,activation = 'softmax'))
    #model.add(K.layers.Dense(26,activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
     
    model.fit(x=train_data, y=train_label, batch_size=32, epochs=2)

    model.evaluate(x=test_data, y=test_label, batch_size=32)