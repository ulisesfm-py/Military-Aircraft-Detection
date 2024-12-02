 # Aircraft Detection using Deep Learning
 ## Overview
 
This project focuses on building a deep learning model for aircraft detection. The model utilizes various deep learning and data handling libraries to process images, train a neural network, and evaluate its performance. The model structure is based on the EfficientNetB3 architecture and employs several callbacks for optimized training.
 
 ## Requirements
 
 To install all the dependencies required for this project, run:
 
 ```bash
 conda create -f environment.yml
 ```
 
 ### Required Libraries
 
 - TensorFlow
 - OpenCV
 - Numpy
 - Pandas
 - Matplotlib
 - Seaborn
 - Scikit-learn
 
 ## Data Preparation
 
 ### 1. Define Paths and Labels
 
 The function `define_paths(dir)` generates file paths and labels for the given directory structure.
 
 ```python
 def define_paths(dir):
     # Your code here
 ```
 
 ### 2. Dataframe Creation
 
 The function `define_df(files, classes)` concatenates file paths and labels into a Pandas DataFrame.
 
 ```python
 def define_df(files, classes):
     # Your code here
 ```
 
 ### 3. Data Splitting
 
 The function `split_data(tr_dir, val_dir=None, ts_dir=None)` handles various formats of input data and splits them into train, validation, and test sets.
 
 ```python
 def split_data(tr_dir, val_dir=None, ts_dir=None):
     # Your code here
 ```
 
 ## Model Architecture
 
 The model is built using the EfficientNetB3 architecture as a base, followed by additional dense layers, dropout, and batch normalization.
 
 ### Model Summary
 
 ```python
 model = Sequential([
     base_model,
     BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
     Dense(256, kernel_regularizer=regularizers.l2(0.016), 
           activity_regularizer=regularizers.l1(0.006), 
           bias_regularizer=regularizers.l1(0.006), 
           activation='relu'),
     Dropout(rate=0.45, seed=123),
     Dense(class_count, activation='softmax')
 ])
 
 model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
 ```
 
 ## Training and Evaluation
 
 The model is trained using the following parameters:
 
 - **Batch Size**: 40
 - **Epochs**: 40
 - **Patience**: 1 (Number of epochs to wait before reducing learning rate if no improvement)
 - **Stop Patience**: 3 (Number of times learning rate can be reduced before stopping training)
 - **Learning Rate Factor**: 0.5 (Factor by which to reduce learning rate)
 - **Threshold**: 0.9 (Accuracy threshold for learning rate adjustments)
 
 ### Training
 
 ```python
 history = model.fit(x=train_gen, 
                     epochs=epochs, 
                     verbose=0, 
                     callbacks=callbacks, 
                     validation_data=valid_gen, 
                     validation_steps=None, 
                     shuffle=False)
 ```
 
 ### Evaluation
 
 ```python
 train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
 valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
 test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)
 ```
 
 ## Visualization
 
 ### Display Sample Images
 
 ```python
 def show_images(gen):
     # Function to display a sample of images from the data generator
 ```
 
 ### Plot Training History
 
 ```python
 def plot_training(hist):
     # Function to plot training and validation accuracy and loss
 ```
 
 ### Confusion Matrix
 
 ```python
 def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
     # Function to plot a confusion matrix
 ```
 
 ## Model Saving and Loading
 
 ### Save Model
 
 ```python
 model.save('model.h5', save_traces=False)
 ```
 
 ### Save Model Weights
 
 ```python
 model.save_weights('weights.h5')
 ```
 
 ## References
 
 1. TensorFlow Documentation
 2. Scikit-learn Documentation
 3. EfficientNet Paper

