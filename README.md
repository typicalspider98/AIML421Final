# README

STUDENT NAME: YE LI  
STUDENT ID: 300665434

## Project Overview 

This project implements a neural network model that combines a custom CNN with a pretrained ResNet18 to classify images of different types of fruits. The dataset includes images of cherries, strawberries, and tomatoes, which are first explored, preprocessed, and augmented before being used to train the model. The model is trained using cross-validation to ensure robustness and better generalizability. 

## Image Paths 

- The training dataset is located in the `./traindata` directory. 
- The test dataset is located in the `./testdata` directory. 

Both directories contain subfolders for each class: `cherry`, `strawberry`, and `tomato`. Make sure that all images are properly sorted into these subfolders for training and testing. 

## Cross-Validation and Model Files 

This project uses K-Fold cross-validation with `n_splits=2`. Therefore, two models will be trained, each corresponding to a different fold of the data. The trained models are saved with the following filenames: 

- `model_fold_1.pth` 
- `model_fold_2.pth` 

Due to submission system limitations, the model files (`.pth`) are not included in the current compressed package as they exceed the file size limit of 100 MB. Instead, the models and the training/testing code have been uploaded to GitHub. Please refer to the provided GitHub link to download the model files.

## Testing the Model 

You can specify which model to use for testing by modifying the model filename in the code. For example, if you want to use the model trained during the first fold, use `model_fold_1.pth`, or for the second fold, use `model_fold_2.pth`.

To change the model for testing, update the following line in the testing code: 

```
model.load_state_dict(torch.load('model_fold_1.pth', map_location=device))
```

Replace `'model_fold_1.pth'` with the filename of the model you want to use. 

## GitHub Link

The complete project, including the model files (`.pth`), training code, and testing code, is available on GitHub. Please visit the following link to access the files:

[GitHub Repository Link](https://github.com/typicalspider98/AIML421Final.git)                 

Ensure that you download the model files from GitHub before attempting to run the testing code.
