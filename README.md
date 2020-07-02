# Indian-Scene-Text-Recognition

The Indian scene text recognition model is developed as part of the work towards [Indian Signboard Translation Project](https://ai4bharat.org/articles/sign-board) by [AI4Bharat](https://ai4bharat.org/). I worked on this project under the mentorship of [Mitesh Khapra](http://www.cse.iitm.ac.in/~miteshk/) and [Pratyush Kumar](http://www.cse.iitm.ac.in/~pratyush/) from IIT Madras.

Indian Signboard Translation  involves 4 modular tasks:
1. *`T1`: Detection:* Detecting bounding boxes containing text in the images
2. *`T2`: Classification:* Classifying the language of the text in the bounding box identifed by `T1`
3. **`T3`: Recognition:** Getting the text from the detected crop by `T1` using the `T2` classified recognition model
4. *`T4`: Translation:* Translating text from `T3` from one Indian language to other Indian language

![Pipeline for sign board translation](../master/Images/Pipeline.jpg)
> Note: `T2`: Classification is not updated in the above picture


# Dataset

[Indian Scene Text Recognition Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset#d3-recognition-dataset) is used.
The `Train` split of the version dataset is used for training. The recognition model is evaluated on all the splits of `D3` and the `Test` split of `D3-V2`


# Model
A Convolutional Recurrent Neural Network Model ([CRNN](https://arxiv.org/pdf/1507.05717v1.pdf)) is used to architect the reconition model for each language individually. The model uses resnet-18 as the feature extractor of images (initialised with pretrained weights on ImageNet). Then the bidirectional gated recurrent units are used to learn from the spatially sequential output of the former CNN part. Finally, a linear output layer is used to classify the character at each sequential step, taking input from the sequential features output of the RNN part.

* Input Image Shape: [200, 50]
* CNN Output Shape: [13, 256]
* RNN Output Shape: [13, 512]
* Linear Output Shape: [13, number of unicode characters]


# Training
The recognition model is trained for 30 epochs for `Tamil` &  `Hindi` and 40 epochs for `Telugu`, `Malayalam` & `Punjabi` with the following hyperpararmeters. The model weights are saved every 10 epochs and you can find them in the [`Models`](../master/Models/) directory

**Hyperparameters: Data Loading**
* batch_size = 64

**Hyperparameters: Model Architecture**
* rnn_hidden_size = 256

**Hyperparameters: Training**
* lr = 0.00081
* weight_decay = 1e-5
* clip_norm = 5
* lr_step_size = 5
* lr_gamma = 0.90

For detailed model architecture and its parameters, check the `Define model` section of the notebook [1-CRNN-Unicode-Tamil.ipynb](../master/1-CRNN-Unicode-Tamil.ipynb)


# Performance
The models after training on the final epoch are used to evaluate the recognition performance. 

Check the recognition accuracies for a range of maximum permitted edit distances of different Indian language recognition models below:

![Tamil Recognition Performance](../master/Images/Recognition-Performance-Tamil.png) 
![Hindi Recognition Performance](../master/Images/Recognition-Performance-Hindi.png) 
![Telugu Recognition Performance](../master/Images/Recognition-Performance-Telugu.png) 
![Malayalam Recognition Performance](../master/Images/Recognition-Performance-Malayalam.png) 
![Punjabi Recognition Performance](../master/Images/Recognition-Performance-Punjabi.png) 


**Incorrectly recognised samples in V2-Testset of Tamil:**

![Misclassifications](../master/Images/Misclassifications-Tamil.png) 


**Code:** 

* Training-Tamil: [CRNN-Unicode-Tamil.ipynb](../master/CRNN-Unicode-Tamil.ipynb)
* Training-Hindi: [CRNN-Unicode-Hindi.ipynb](../master/CRNN-Unicode-Hindi.ipynb)
* Training-Telugu: [CRNN-Unicode-Telugu.ipynb](../master/CRNN-Unicode-Telugu.ipynb)
* Training-Malayalam: [CRNN-Unicode-Malayalam.ipynb](../master/CRNN-Unicode-Malayalam.ipynb)
* Training-Punjabi: [CRNN-Unicode-Punjabi.ipynb](../master/CRNN-Unicode-Punjabi.ipynb)

* Evaluation-Tamil: [Test-CRNN-Unicode-Tamil.ipynb](../master/Test-CRNN-Unicode-Tamil.ipynb)
* Evaluation-Hindi: [Test-CRNN-Unicode-Hindi.ipynb](../master/Test-CRNN-Unicode-Hindi.ipynb)
* Evaluation-Telugu: [Test-CRNN-Unicode-Telugu.ipynb](../master/Test-CRNN-Unicode-Telugu.ipynb)
* Evaluation-Malayalam: [Test-CRNN-Unicode-Malayalam.ipynb](../master/Test-CRNN-Unicode-Malayalam.ipynb)
* Evaluation-Punjabi: [Test-CRNN-Unicode-Punjabi.ipynb](../master/Test-CRNN-Unicode-Punjabi.ipynb)

### Related Links:
1. [Indian Signboard Translation Project](https://ai4bharat.org/articles/sign-board)
2. [Indian Scene Text Dataset](https://github.com/GokulKarthik/Indian-Scene-Text-Dataset)
3. [Indian Scene Text Detection](https://github.com/GokulKarthik/Indian-Scene-Text-Detection)
4. [Indian Scene Text Classification](https://github.com/GokulKarthik/Indian-Scene-Text-Classification)
5. [Indian Scene Text Recognition](https://github.com/GokulKarthik/Indian-Scene-Text-Recognition)

### References:
1. https://arxiv.org/pdf/1507.05717v1.pdf
2. https://arxiv.org/pdf/1512.03385.pdf
3. https://github.com/GokulKarthik/Deep-Learning-Projects.pytorch
4. https://github.com/carnotaur/crnn-tutorial/
