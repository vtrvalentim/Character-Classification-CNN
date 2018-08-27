# Character-Classification-CNN

This package contains a data set with over 300k hand written characters and a convolutional neural network that classifies 
these carachters with 99.6% test accuracy

How to run this algorithm:
1. Open ccConvNet.py
2. Comment/Uncomment one of the two lines depending on your usage case
from module1 import CreateInput -- use module 1 to run the CNN for the entire data set (might crash your computer)
from module2 import CreateInput -- use module 2 to run the CNN for a smaller data set (5 characters, runs much faster)
3. Run ccConvNet.py
