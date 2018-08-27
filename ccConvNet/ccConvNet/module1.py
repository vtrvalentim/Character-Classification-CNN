import cv2 as cv2
import glob as glob
import numpy as np

def Imgset2Vector(path, label_value):

    #inputs: file path of images e.g. "../DATA SET/A/*.png"
    #        label vector for that class; dim = 26x1

    #outputs: numpy array containing RGB values of all images on file path; dim = m x 128 x 128 x 3 
    #         numpy array with the corresponding label; dim = 26 x m

    temp_list = []

    file_path = glob.glob(path)

    for file in file_path:
        image = cv2.imread(file,1)
        temp_list.append(image)
    data_array = np.array(temp_list)
    temp_list.clear()

    print(label_value.shape)
    labels = np.repeat(label_value,data_array.shape[0],axis=1)
    print(labels.shape)

    return data_array, labels

def DataShuffle(data,label):

    #inputs: numpy array with data
    #        numpy array with respective labels
    #outputs: none. Function just shuffles the data.

    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(label)

    pass

def DeclarePath():

    #output: list with file paths for all img sets

    path_list = []

    path_list.append("../DATA/A/*.png")
    path_list.append("../DATA/B/*.png")
    path_list.append("../DATA/C/*.png")
    path_list.append("../DATA/D/*.png")
    path_list.append("../DATA/E/*.png")
    path_list.append("../DATA/F/*.png")
    path_list.append("../DATA/G/*.png")
    path_list.append("../DATA/H/*.png")
    path_list.append("../DATA/I/*.png")
    path_list.append("../DATA/J/*.png")
    path_list.append("../DATA/K/*.png")
    path_list.append("../DATA/L/*.png")
    path_list.append("../DATA/M/*.png")
    path_list.append("../DATA/N/*.png")
    path_list.append("../DATA/O/*.png")
    path_list.append("../DATA/P/*.png")
    path_list.append("../DATA/Q/*.png")
    path_list.append("../DATA/R/*.png")
    path_list.append("../DATA/S/*.png")
    path_list.append("../DATA/T/*.png")
    path_list.append("../DATA/U/*.png")
    path_list.append("../DATA/V/*.png")
    path_list.append("../DATA/W/*.png")
    path_list.append("../DATA/X/*.png")
    path_list.append("../DATA/Y/*.png")
    path_list.append("../DATA/Z/*.png")

    return path_list

def DeclareLabels():

    #output: matrix with labels for each class at each column

    label_matrix = np.identity(26)

    return label_matrix

def SplitData(data_set, labels):

    #input: Data set with all data points, label set with respective labels
    #output: Train data and label set, Test data and label set

    train_size = int(0.9 * data_set.shape[0])

    train_data = data_set[0:train_size]
    train_label = labels[0:train_size]

    test_data = data_set[train_size:]
    test_label = labels[train_size:]

    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)

    return train_data, train_label, test_data, test_label

def CreateInput():

    #this function is to be called in the main program to generate the input sets for the NN 
    
    path_list = DeclarePath()
    label_matrix = DeclareLabels()

    data_list = np.zeros((1,128,128,3))
    label_list = np.zeros((26,1))

    for i in range(0,25):
        data_array, label_array = Imgset2Vector(path_list[i],label_matrix[:,[i]])
        data_list = np.concatenate((data_list,data_array))
        label_list = np.concatenate((label_list,label_array), axis=1)

    data_list = data_list[1:]
    label_list = np.delete(label_list,0,1)

    label_list = label_list.T

    DataShuffle(data_list,label_list)

    print(data_list.shape)
    print(label_list.shape)
    print("")

    train_data, train_label, test_data, test_label = SplitData(data_list,label_list)

    return train_data, train_label, test_data, test_label

