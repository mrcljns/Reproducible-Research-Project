from train_and_predict import train_and_predict, MODELS_MAPPING
import pandas as pd
import os

COMBINATIONS = {"breast_cancer_X.csv": "breast_cancer_y.csv",
                "chronic_kidney_disease_X_mean_imputted.csv": "chronic_kidney_disease_y.csv",
                "chronic_kidney_disease_X_median_imputted.csv": "chronic_kidney_disease_y.csv",
                "chronic_kidney_disease_X_missing_dropped.csv": "chronic_kidney_disease_y_missing_dropped.csv",
                "covid_19_X.csv": "covid_19_y.csv",
                "cryotherapy_X.csv":"cryotherapy_y.csv",
                "hepatitis_X_mean_imputted.csv": "hepatitis_y.csv",
                "hepatitis_X_median_imputted.csv": "hepatitis_y.csv",
                "hepatitis_X_missing_dropped.csv": "hepatitis_y_missing_dropped.csv",
                "immunotherapy_X.csv": "immunotherapy_y.csv"}

def main():
    results_df = pd.DataFrame(columns = ["dataset", "model", "train_time", "eval_time", "train_acc", "eval_acc", "train_f1", "eval_f1"])
    for x_path, y_path in COMBINATIONS.items():
        full_x_path = os.path.join("../Data/Cleaned", x_path)
        full_y_path = os.path.join("../Data/Cleaned", y_path)
        for model_type in MODELS_MAPPING:
            print(x_path, y_path, model_type)
            train_time, eval_time, train_accuracy, eval_accuracy, train_f1, eval_f1 = train_and_predict(full_x_path, full_y_path, MODELS_MAPPING[model_type])
            
            new_row = {"dataset": x_path.split(".")[0].replace("_X", ""),
                       "model": model_type,
                       "train_time": train_time,
                       "eval_time": eval_time,
                       "train_acc": train_accuracy,
                       "eval_acc": eval_accuracy,
                       "train_f1": train_f1,
                       "eval_f1": eval_f1}
            
            results_df.loc[len(results_df)] = new_row
    results_df.to_csv("../Data/results.csv")
    
if __name__ == "__main__":
    main()
            
