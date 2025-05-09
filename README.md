**Iris Classification: Decision Tree & Random Forest
Project Overview**
This project demonstrates classification of the classic Iris flower dataset using machine learning. It focuses on building and evaluating two models: a Decision Tree Classifier and a Random Forest Classifier implemented with scikit-learn.
The workflow includes loading the dataset, splitting it into training and test sets, training each model, and evaluating their performance. Accuracy and confusion matrix results are reported for each model. The notebook also visualizes the decision tree structure to illustrate the model’s logic.

**Features**
•	Classic Iris flower dataset (3 classes) included via scikit-learn
•	Decision Tree classifier implementation and evaluation
•	Random Forest classifier implementation and evaluation
•	Model performance metrics: accuracy and confusion matrix
•	Decision tree structure visualization using Matplotlib

**Installation**
1.	Ensure you have Python 3.x installed on your system.
2.	Install the required Python libraries by running:

pip install numpy pandas scikit-learn matplotlib

3.	Clone or download this repository to your local machine.
4.	Launch Jupyter Notebook (or JupyterLab) and open Iris Classification (Decision Tree and Random Forest).ipynb.

**Example Usage**
Open the Jupyter Notebook and run all cells. The notebook will automatically load the Iris dataset, train the Decision Tree and Random Forest models, and display the accuracy and confusion matrix results. For example, the Decision Tree is trained in the notebook with code similar to the following:

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=50
)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

**Dataset**
This project uses the classic Iris flower dataset, which contains 150 samples of iris flowers with four features (sepal length, sepal width, petal length, petal width) and three species labels. The dataset is loaded directly from scikit-learn via load_iris(), so no separate download is needed. It is a well-known benchmark dataset in machine learning.

**Contributing**
This is a personal project. Contributions are welcome – feel free to fork the repository and submit pull requests for improvements or new features. If you encounter any issues, please open an issue to report them.

