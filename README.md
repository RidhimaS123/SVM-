# SVM-
This project applies Support Vector Regression (SVR) to a character recognition dataset. It uses a pipeline with scaling and hyperparameter tuning via GridSearchCV, and evaluates model performance over 10 randomized train/test splits.
 Dataset
The dataset used is a character recognition dataset where each sample is a character described by 16 numerical features. The letter column represents the target class, and the rest are feature columns like x-box, y-box, width, etc.

Note: The dataset file is expected to be located at:

/content/drive/MyDrive/Classroom/SVM
You can replace the path with a local one if needed.

 Objectives
Train and optimize an SVR model using Grid Search.

Evaluate model accuracy using rounded predictions.

Simulate and visualize convergence (approximate learning curve).

Save results for further analysis.

Technologies Used
Python

Pandas & NumPy

Scikit-learn (SVR, Pipelines, LabelEncoder, GridSearchCV, etc.)

Matplotlib (for plotting convergence, optional)

 Hyperparameter Tuning
A grid search is performed across the following parameters:

param_grid = {
    'svr__kernel': ['linear', 'rbf', 'poly'],
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.5]
}
Each configuration is evaluated using 3-fold cross-validation.

Repeated Evaluation
The process is repeated 10 times with different random splits. For each iteration, the best estimator is used to predict on the test set, and results are stored:

Accuracy (rounded SVR output)

Best kernel, C, and epsilon values

Convergence trend (simulated by incrementally training on increasing subsets of the training data)

 Output
Results Table

Sample	Accuracy (%)	Kernel	C	Epsilon
S1	91.23	rbf	10	0.1
S2	90.85	poly	1	0.1
...	...	...	...	...
This table is also saved as a CSV:


svm_results.csv
Installation
Install required libraries:

pip install pandas numpy scikit-learn matplotlib
How to Run
Clone the repository:

git clone https://github.com/yourusername/svm-character-recognition.git
cd svm-character-recognition
Place your dataset in the path specified in the script (or change the path).

Run the script:

python svm_optimization.py
(Optional) Visualize convergence for the best-performing model using:

import matplotlib.pyplot as plt
plt.plot(best_convergence)
plt.xlabel("Percentage of Training Data")
plt.ylabel("Accuracy (%)")
plt.title("Simulated Convergence Curve")
plt.show()
