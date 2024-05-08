from utils import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle as pkl

MODELS_MAPPING = {"rf": RandomForestClassifier(random_state=42),
                  "knn": KNeighborsClassifier(random_state=42),
                  "svm": SVC(random_state=42),
                  "dt": DecisionTreeClassifier(random_state=42)} 
TEST_SPLIT_FRACTION = 0.2

def define_model(model_type):
    return MODELS_MAPPING[model_type]

def optimize_hyperparams(x_path, y_path, model_type, k):
    X, y  = load_dataset(x_path, y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_FRACTION, random_state=42)
    model = define_model(model_type)
    if model_type == "rf":
        params = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
            }
    elif model_type == "knn":
        params = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
            }
    elif model_type == "svm":
        params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
            }
    elif model_type == "dt":
        params = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
            }
    else:
        raise ValueError("Invalid model type.")
    clf = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=k, n_jobs=-1)
    clf.fit(X_train, y_train)
    # Save the best params to a python dict
    with open(f'../best_hyperparams/{model_type}_best.pickle', 'wb') as handle:
        pkl.dump(clf.best_params_, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    for model_type in MODELS_MAPPING.keys():
        model = optimize_hyperparams(x_path="../Data/Cleaned/breast_cancer_X.csv",
                                    y_path="../Data/Cleaned/breast_cancer_y.csv",
                                    model_type=model_type,
                                    k=5) 