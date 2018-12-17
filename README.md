# Automated Diagnostic Classification of Pediatric MR Angiography Using Deep Learning

NOTE: The pediatric MRA dataset we used for training and testing our models is not publicly available, as it is patient information.

Cerebrovascular diseases are an important cause of mortality and morbidity within the pediatric population, and effective treatment depends on accurate and efficient diagnosis. Magnetic resonance angiography (MRA) is a noninvasive technology that allows visualization of the vasculature of the pediatric brain without necessitating exposure to radiation and is an important diagnostic imaging tool used for evaluating suspected cerebrovascular disease in children. However, due to complex developmental changes of the brain and skull, interpretation of pediatric MRA remains an enormous challenge for most clinicians and a source of significant burden when life-threatening events require immediate diagnosis and decision-making. Therefore, there is an urgent need for a more efficient and accurate diagnostic decision support for pediatric MRA. We applied deep-learning methodologies to develop a classifier model to distinguish normal versus abnormal MRA. Our model relies on a Residual Network (ResNet) architecture pre-trained on ImageNet. We also include in our repository a logistic regression model as a baseline.

## Getting Started

Clone the repository to your local machine. The main scripts you'll work with are `logistic_regression.py` as well as the scripts contained within the `CS230SampleCode` directory.

For these scripts, you will need a version of Python 3.

## Running the Tests

To run our logistic regression baseline model, simply enter in your terminal command line, 
```
python3 logistic_regression.py
``` 
It is written into the script to pull the images from the server. For the purposes of our dataset, we added a bit of code that saves the matrix that represents our entire dataset locally as a .txt file, which is the file we use to train and test our convolutional neural net (CNN) models, so run logistic regression first before moving onto CNNs. You can go in and edit the data pulling to feed in your own dataset.

To run our convolutional neural net models, first cd into CS230SampleCode. Depending on whether you want to use the default CS230SampleCode or pull a pre-trained ResNet18 or ResNet50, open `train.py` and go to line 188 to comment out the model you don't want and uncomment the model you do want. When using the ResNet models, be sure to uncomment the bit of code immediately after that adds in a few layers to the end of the pre-trained model. Then, cd into model and open `data_loader.py` to edit the dataset that you would like to feed into the model. 

Once the setup for the CNN model is complete, enter into your terminal command line, 
```
python3 train.py --model_dir experiments/base_model
``` 
You can also run a hyperparameter search of the learning rate after training your model by entering into your terminal command line, 
```
python3 search_hyperparams.py --parent_dir experiments/learning_rate
``` 
By default it tests learning rates of .01, .001, and .0001. You can display the results of the different hyperparameters by running 
```
python3 synthesize_results.py --parent_dir experiments/learning_rate
```

## Authors
* **Yong-hun Kim**
* **Ye Akira Wang**
* **Dennis Chang**
