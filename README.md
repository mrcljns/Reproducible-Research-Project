# Reproducible-Research-Project
Our project is centered on replicating and expanding upon the machine learning research detailed in the Asian Journal of Computer and Information Systems (AJCIS), specifically the article titled *"Utilisation of Machine Learning Techniques in Testing and Training of Different Medical Datasets"*. To reproduce the results obtained in this project, use the ```conda create --name <env> --file requirements.txt``` command to recreate the anaconda environment and run the scripts in the following order:
1. ```scripts/data_cleaning_eda.py```
   - clean and transform the data from the ```Data/Raw``` folder,
   - save preprocessed datasets to the ```Data/Cleaned``` folder and visualizations to the ```EDA``` folder
2. ```scripts/run_gridsearch.py```
   - obtain the best hyperparameters for each model
   - save hyperparameters in *.pickle* files in the ```Data/Best_hyperparams``` folder
3. ```scripts/train_all_model.py```
   - train all models with the best hyperparameters on each of the datasets from the ```Data/Cleaned``` folder
   - save results in the ```Data``` folder
