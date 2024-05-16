from utils import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import numpy as np
import pickle as pkl

def load_hyperparams(model_type):
    with open(f'../Data/Best_hyperparams/{model_type}_best.pickle', 'rb') as handle:
        params = pkl.load(handle)
        return params

MODELS_MAPPING = {
    "rf": RandomForestClassifier(**load_hyperparams("rf"), n_jobs=-1, random_state=42),
    "knn": KNeighborsClassifier(**load_hyperparams("knn"), n_jobs=-1),
    "svm": SVC(**load_hyperparams("svm"), random_state=42),
    "dt": DecisionTreeClassifier(**load_hyperparams("dt"), random_state=42)
    } 
TEST_SPLIT_FRACTION = 0.2

def define_model(model_type):
    return MODELS_MAPPING[model_type]

def calculate_metrics(y_train, y_test, y_pred_train, y_pred_eval, case='binary'):
    train_accuracy = accuracy_score(y_train, y_pred_train)
    eval_accuracy = accuracy_score(y_test, y_pred_eval)

    if case == 'binary':
        train_f1_score = f1_score(y_train, y_pred_train)
        eval_f1_score = f1_score(y_test, y_pred_eval)
    elif case == 'multiclass':
        train_f1_score = f1_score(y_train, y_pred_train, average='weighted')
        eval_f1_score = f1_score(y_test, y_pred_eval, average='weighted')
    else:
        raise ValueError("Invalid value for 'case'. Please specify either 'binary' or 'multiclass'.")

    return train_accuracy, train_f1_score, eval_accuracy, eval_f1_score

def train_and_predict(x_path, y_path, model):
    X, y = load_dataset(x_path, y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_FRACTION, random_state=42)
    training_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - training_start
    y_pred_train = model.predict(X_train)
    
    eval_start = time.time()
    y_pred_eval = model.predict(X_test)
    eval_time = time.time() - eval_start
    
    if np.unique(y).size > 2:
        case="multiclass"
    else:
        case="binary"
    
    train_accuracy, train_f1_score, eval_accuracy, eval_f1_score = calculate_metrics(y_train, y_test, y_pred_train, y_pred_eval, case=case)
    
    return training_time, eval_time, train_accuracy, eval_accuracy, train_f1_score, eval_f1_score

if __name__ == "__main__": 
    model = define_model("rf")
    training_time, testing_time, train_acc, eval_acc = train_and_predict(x_path="../Data/Cleaned/breast_cancer_X.csv",
                                                                         y_path="../Data/Cleaned/breast_cancer_y.csv",
                                                                         model=model,
                                                                         save_path="test")
    print(training_time, testing_time, train_acc, eval_acc)