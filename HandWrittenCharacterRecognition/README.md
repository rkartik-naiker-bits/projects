In this project, we have implemented character recognizer using Convolutional Neural Network and analysed the effect of dropout layers. Theory explains that dropout layers are part of neural networks to prevent over fitting of the nodes and thereby removing codependency among the neurons while training. In simpler terms, using dropout layers, some neurons(features) from the networks are dropped randomly, and thereby allows the network to give equal weights/importance to all the features while learning rather than focusing on one particular feature.
We have also compared the performance of 3 API’s by comparing their accuracies for the trained model. Following are the API’s used

•	Keras
•	MxNet
•	PyTorch
 
Dataset - We have used the EMNIST dataset which consists of handwritten character digits converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset.
We implemented 3 different CNN models using each API. Each model was tested with 3 different sets of parameters, with and without dropout. 

The details of the CNN architectures used and the results are available in the CNN_Architecture_and_Results.xlx file

In terms of accuracy, time taken and overall performance:
Keras > PyTorch > MxNet

Overall, Keras gave the best performance in terms of accuracy, and time taken closely followed by PyTorch. With dropout, the results in PyTorch were marginally better than without dropout.

