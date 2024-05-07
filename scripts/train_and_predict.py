from utils import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

MODELS_MAPPING = {"rf": RandomForestClassifier(random_state=42),
                  "knn": KNeighborsClassifier(random_state=42),
                  "svm": SVC(random_state=42),
                  "ds": DecisionTreeClassifier(random_state=42)} 
TEST_SPLIT_FRACTION = 0.2

def define_model(model_type):
    return MODELS_MAPPING[model_type]

def train_and_predict(x_path, y_path, model, save_path):
    X, y  = load_dataset(x_path, y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_FRACTION, random_state=42)
    training_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - training_start
    
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    eval_start = time.time()
    y_pred_eval = model.predict(X_test)
    eval_time = time.time() - eval_start
    eval_accuracy = accuracy_score(y_test, y_pred_eval)
    
    return training_time, eval_time, train_accuracy, eval_accuracy

if __name__ == "__main__": 
    model = define_model("rf")
    training_time, testing_time, train_acc, eval_acc = train_and_predict(x_path="../Data/Cleaned/breast_cancer_X.csv",
                                                                         y_path="../Data/Cleaned/breast_cancer_y.csv",
                                                                         model=model,
                                                                         save_path="test")
    print(training_time, testing_time, train_acc, eval_acc)