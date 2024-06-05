# Reproducible-Research-Project
To recreate the results obtained in the project, run the scripts in the following order:
1. ```scripts/data_cleaning_eda.py``` - clean and transform the data from the ```Data/Raw``` folder, save preprocessed datasets to the ```Data/Cleaned``` folder and visualizations to the ```EDA``` folder
2. ```scripts/run_gridsearch.py``` - obtain the best hyperparameters for each model and save them in *.pickle* files in the ```Data/Best_hyperparams``` folder
3. ```scripts/train_all_model.py``` - train all models with the best hyperparameters on each of the datasets from the ```Data/Cleaned``` folder and save results in the ```Data``` folder
