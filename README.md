# Enhancing-Music-Release-Year-Prediction

Modified the nn.py to accommodate a classification task where song labels were predicted based on their release periods. 
Exploration of hyperparameters, cross-validation, and wrapper functions showcased understanding of neural networks.


## Part 1: Neural Network Implementation and Training (Regression Task) 

**Part 1.A: Neural Network Implementation**

Successfully implemented the neural network using the functions provided in nn.py.<br>
Utilized mini-batch gradient descent with the Mean Squared Error (MSE) loss function. <br>
ReLU activation functions were applied to all hidden layers. You can find code in the file nn1.py.

**Part 1.B: Training and Dev Set Loss Plot **

Visualized data by plotting the train and dev set loss after each epoch for batch sizes 32 and 64. <br>
These plots help us understand the network's learning progress and performance over time. <br> 

You can find the data in train_32.png, dev_32.png, train_64.png, and dev_64.png.

## Part 2: Evaluating Performance on Test Data 

Evaluated the neural network's performance on the test data provided in test.csv. <br>
Explored various hyperparameters and possibly used cross-validation techniques to fine-tune the model. <br>
Predictions were submitted in the required <roll_number>.csv format, and created a detailed summary of hyperparameters in part_2.csv.

Code for this part can be found in nn2.py.

## Part 3 : Feature Selection 

Created a new feature set based on the original 90 features, while aiming for a smaller size and comparable performance. 
Involved identifying potentially useful features and possibly combining them to create efficient features.

**Part 3.A: Feature Selection Implementation**

Implemented feature selection in the nn3.py file. Created a new feature set with a size strictly smaller than 90 while ensuring that the performance using this subset of features is at least as good as using the complete feature set and applied various techniques and methods to achieve this task.

Code for feature selection can be found in nn3.py.

**Part 3.B: Hyperparameter Summary**

To provide a comprehensive understanding of feature selection process, submitted part_3.csv. This CSV file contains the details of any new hyperparameters or variables introduced in implementation for feature selection.

**Part 3.C: Selected Features**

Provided valuable insights by submitting features.csv. This CSV file includes a list of the selected features from the original set of 90 features. Each feature in this new set could be a combination of features from the original set. Successfully managed the transformation and ensured that the total number of features is less than 90.
 
**Classification Task**

As an extra mile, embraced the classification task where we predicted song labels based on their release periods. Shown remarkable adaptability by modifying the nn.py file to accommodate this task. 

Code for the extra credit classification task can be found in nn_extra_credit.py.
 


Assignment's submission requirements:

- nn_1.py: Your implementation of the neural network for Part 1.A
- train_32.png: Plot showing train set loss for batch size 32
- dev_32.png: Plot showing dev set loss for batch size 32
- train_64.png: Plot showing train set loss for batch size 64
- dev_64.png: Plot showing dev set loss for batch size 64
- nn_2.py: Your code for evaluating performance on test data in Part 2
- part_2.csv: Detailed summary of hyperparameters for Part 2
- nn_3.py: Your code for feature selection in Part 3
- part_3.csv: Detailed summary of hyperparameters for Part 3
- features.csv: List of selected features for Part 3
- nn_extra_credit.py: Your code for the extra credit classification task
