# Library to directly download datasets from UC Irvine Machine Learning Repo
from ucimlrepo import fetch_ucirepo
# Library to handle and manipulate data
import pandas as pd 
# Set option to suppress chained assignment warning
pd.options.mode.chained_assignment = None  
# Library for numerical operations
import numpy as np 
# Libraries for data visualization
import seaborn as sns 
import matplotlib.pyplot as plt 
# Library for mathematical operations
import math 
# Library for operating system-related functions
import os

def load_datasets():
    """
    Load various datasets including Covid-19, Hepatitis, Chronic Kidney Disease,
    Breast Cancer, Cryotherapy, and Immunotherapy datasets.
    """
    # Dictionary to store datasets
    datasets = {}

    # Covid 19 dataset
    covid_19_surveillance = fetch_ucirepo(id=567)
    datasets['covid_19_X'] = covid_19_surveillance.data.features
    datasets['covid_19_y'] = covid_19_surveillance.data.targets

    # Hepatisis dataset
    hepatitis = fetch_ucirepo(id=46)
    datasets['hepatitis_X'] = hepatitis.data.features
    datasets['hepatitis_y'] = hepatitis.data.targets

    # Chronic kidney disease dataset
    chronic_kidney_disease = fetch_ucirepo(id=336)
    datasets['chronic_kidney_disease_X'] = chronic_kidney_disease.data.features
    datasets['chronic_kidney_disease_y'] = chronic_kidney_disease.data.targets

    # Breast cancer dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    datasets['breast_cancer_X'] = breast_cancer_wisconsin_diagnostic.data.features
    datasets['breast_cancer_y'] = breast_cancer_wisconsin_diagnostic.data.targets

    # Cryotherapy dataset
    Cryotherapy = pd.read_excel("Data//Raw//Cryotherapy.xlsx")
    datasets['cryotherapy_X'] = Cryotherapy.drop(columns=['Result_of_Treatment'])
    datasets['cryotherapy_y'] = Cryotherapy.Result_of_Treatment

    # Immunotherapy dataset
    Immunotherapy = pd.read_excel("Data//Raw//Immunotherapy.xlsx")
    datasets['immunotherapy_X'] = Immunotherapy.drop(columns=['Result_of_Treatment'])
    datasets['immunotherapy_y'] = Immunotherapy.Result_of_Treatment

    return datasets


def is_complete(row):
    """
    Check if a row is complete (contains no missing values).
    """
    return row.notnull().all()

def preprocess_data(X, Y):
    """
    Preprocesses the input data:
    1. Imputes missing values with the mean of each column.
    2. Imputes missing values with the median of each column.
    3. Drops rows with missing values from both the explanatory (X) and target (Y) variables.
    """
    
    # Calculate the mean and median of each column in X
    column_means = X.mean()
    column_medians = X.median()
    
    # Create copies of X for mean and median imputation
    X_mean_imputted = X.fillna(column_means)
    X_median_imputted = X.fillna(column_medians)
    
    # Create copies of X and Y for dropping missing data
    X_missing_dropped = X.dropna()
    Y_missing_dropped = Y.loc[X_missing_dropped.index]
    
    return X_mean_imputted, X_median_imputted, X_missing_dropped, Y_missing_dropped

def plot_target_variable_distribution(target_variable, dataset_name):
    """
    Plots the distribution of the target variable and saves the plot as an image.
    """
    # Plot the target variable distribution
    plt.figure(figsize=(4, 3))
    target_variable.value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Define the directory and file path to save the plot
    directory = os.path.join(os.getcwd(), 'EDA')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{dataset_name}_target.png"
    filepath = os.path.join(directory, filename)

    # Save the plot as an image
    plt.savefig(filepath)
    # plt.show()
    # plt.close()
    print(f"Target variable saved as '{filepath}'")

def plot_variable_distribution(dataframe, dataset_name):
    """
    Plots the distribution of each variable in the input DataFrame and saves the plot as an image.
    """
    # Define the number of rows based on the number of columns and keeping ncols=5
    ncols = 5
    nrows = math.ceil(len(dataframe.columns) / ncols)
    
    # Get the binary columns starting with "is_"
    binary_columns = [col for col in dataframe.columns if col.startswith('is_')]
    
    # Set up the figure and axes
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows))
    
    # Flatten the axs array to iterate over it easily
    axs = axs.flatten()
    
    # Iterate over each column and plot its distribution
    for i, col in enumerate(dataframe.columns):
        if col in binary_columns:
            # Count the frequency of each category
            counts = dataframe[col].value_counts().sort_index()
            # Plot the bars next to each other
            counts.plot(kind='bar', ax=axs[i], color=['skyblue', 'lightcoral'])
            axs[i].set_title(f'Distribution of {col}')
            axs[i].set_ylabel('Frequency')
            # Force the x ticks to be integers
            axs[i].set_xticks(range(len(counts)))
            axs[i].set_xticklabels(counts.index.astype(int), rotation=0)  # Set x tick labels as integers
        else:
            # Plot the histogram for non-binary columns
            axs[i].hist(dataframe[col], bins=20, color='skyblue', edgecolor='black')
            axs[i].set_title(f'Distribution of {col}')
            axs[i].set_xlabel(col)
            axs[i].set_ylabel('Frequency')
    
    # Hide any empty subplots
    for ax in axs[len(dataframe.columns):]:
        ax.axis('off')
    
    # Adjust layout and show the plot
    plt.tight_layout()

    # Define the directory and file path to save the plot
    directory = os.path.join(os.getcwd(), 'EDA')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{dataset_name}_variable_distribution.png"
    filepath = os.path.join(directory, filename)

    # Save the plot as an image
    plt.savefig(filepath)
    # plt.show()
    # plt.close()
    print(f"Explanatory variables distribution plot saved as '{filepath}'")

def plot_correlation_matrix(X, y=None, dataset_name=None):
    """
    Plots the correlation matrix for explanatory variables and the target variable (if provided)
    and saves the plot as an image.
    """
    # Concatenate X and y if y is provided
    if y is not None:
        data = pd.concat([X, y], axis=1)
    else:
        data = X
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create a mask to hide the upper triangle
    corr_matrix_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot correlation matrix showing only the lower triangle
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr_matrix, mask=corr_matrix_mask, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5, annot_kws={"size": 8})    
    
    # Set plot title
    if y is not None:
        plt.title('Correlation Matrix for Explanatory and Target Variables')
    else:
        plt.title('Correlation Matrix for Explanatory Variables')

    # Define the directory and file path to save the plot
    directory = os.path.join(os.getcwd(), 'EDA')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{dataset_name}_corr_matrix.png"
    filepath = os.path.join(directory, filename)

    # Save the plot as an image
    plt.savefig(filepath)
    # plt.show()
    # plt.close()
    print(f"Correlation matrix plot saved as '{filepath}'")

def covid_19_prep(covid_19_X, covid_19_y):
    """
    Preprocess the Covid-19 dataset and generate plots for variable distributions and correlation matrix.
    """
    # Replace '+' with 1 and '-' with 0 in each column
    covid_19_X = covid_19_X.replace({'+': 1, '-': 0})

    # Convert columns to numeric
    covid_19_X = covid_19_X.apply(pd.to_numeric)

    # Prefix columns with 'is_'
    covid_19_X.columns = ['is_' + col for col in covid_19_X.columns]

    # Plot the variable distribution
    plot_variable_distribution(covid_19_X, "covid_19_X")

    # Plot the correlation matrix
    plot_correlation_matrix(covid_19_X, dataset_name="covid_19_X")

    # Plot the target variable distribution
    plot_target_variable_distribution(covid_19_y, "covid_19_y")

    return covid_19_X, covid_19_y


def hepatitis_prep(hepatitis_X, hepatitis_y):
    """
    Preprocess the Hepatitis dataset and generate plots for variable distributions, correlation matrix,
    and perform mean/median imputation and dropping missing rows.
    """
    # List of columns to apply binary remapping
    columns_to_process = ['Steroid', 'Sex', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big',
                          'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology']
    
    for col in columns_to_process:
        if col == 'Sex':
            hepatitis_X['is_female'] = hepatitis_X[col] - 1
        else:
            # Apply the binary remapping: convert to binary and rename the column
            hepatitis_X[f'is_{col.lower()}'] = hepatitis_X[col] - 1
        # Drop the original column
        hepatitis_X.drop(columns=[col], inplace=True)

    # Also apply binary remapping to target column
    hepatitis_y['is_alive'] = hepatitis_y['Class'] - 1
    hepatitis_y.drop(columns=['Class'], inplace=True)

    # Plot the distribution of each explanatory variable (X)
    plot_variable_distribution(hepatitis_X, "hepatitis_X")

    # Plot the correlation matrix for explanatory variables (X) and the target variable (y)
    plot_correlation_matrix(hepatitis_X, hepatitis_y, "hepatitis_X_y")

    # Plot the distribution of the target variable (y)
    plot_target_variable_distribution(hepatitis_y, "hepatitis_y")

    # Preprocess the data: mean imputation, median imputation, and dropping missing rows
    hepatitis_X_mean_imputted, hepatitis_X_median_imputted, hepatitis_X_missing_dropped, hepatitis_y_missing_dropped = preprocess_data(hepatitis_X, hepatitis_y)

    return hepatitis_X_mean_imputted, hepatitis_X_median_imputted, hepatitis_X_missing_dropped, hepatitis_y_missing_dropped

def chronic_kidney_disease_prep(chronic_kidney_disease_X, chronic_kidney_disease_y):
    """
    Preprocess the Chronic Kidney Disease dataset and generate plots for variable distributions, 
    correlation matrix, and perform mean/median imputation and dropping missing rows.
    """
    # Apply binary mapping to exp variables

    # Mapping and dropping columns
    mappings = {
        'rbc': {'normal': 1, 'abnormal': 0},
        'pc': {'normal': 1, 'abnormal': 0},
        'pcc': {'present': 1, 'notpresent': 0},
        'ba': {'present': 1, 'notpresent': 0},
        'htn': {'yes': 1, 'no': 0},
        'dm': {'yes': 1, 'no': 0, '\tno': 0},
        'cad': {'yes': 1, 'no': 0},
        'appet': {'good': 1, 'poor': 0},
        'pe': {'yes': 1, 'no': 0},
        'ane': {'yes': 1, 'no': 0}
    }
    
    for col, mapping in mappings.items():
        chronic_kidney_disease_X[f'is_{col}'] = pd.to_numeric(chronic_kidney_disease_X[col].map(mapping))
        chronic_kidney_disease_X.drop(columns=[col], inplace=True)
    
    # Apply binary mapping to target variable 
    # class = ckd: 248, notckd: 150, ckd\t: 2
    chronic_kidney_disease_y['is_chronic_kidney_disease'] = pd.to_numeric(chronic_kidney_disease_y['class'].map({'ckd': 1, 'notckd': 0, 'ckd\t': 0}))
    chronic_kidney_disease_y.drop(columns=['class'], inplace=True)

    # Plot the distribution of each explanatory variable (X)
    plot_variable_distribution(chronic_kidney_disease_X, "chronic_kidney_disease_X")

    # Plot the correlation matrix for explanatory variables (X) and the target variable (y)
    plot_correlation_matrix(chronic_kidney_disease_X, chronic_kidney_disease_y, "chronic_kidney_disease_X_y")

    # Plot the distribution of the target variable (y)
    plot_target_variable_distribution(chronic_kidney_disease_y, "chronic_kidney_disease_y")

    # Preprocess the data: mean imputation, median imputation, and dropping missing rows
    chronic_kidney_disease_X_mean_imputted, chronic_kidney_disease_X_median_imputted, chronic_kidney_disease_X_missing_dropped, chronic_kidney_disease_y_missing_dropped = preprocess_data(chronic_kidney_disease_X, chronic_kidney_disease_y)

    return chronic_kidney_disease_X_mean_imputted, chronic_kidney_disease_X_median_imputted, chronic_kidney_disease_X_missing_dropped, chronic_kidney_disease_y_missing_dropped

def breast_cancer_prep(breast_cancer_X, breast_cancer_y):
    """
    Preprocess the Breast Cancer dataset and generate plots for variable distributions, 
    correlation matrix, and target variable distribution.
    """
    # Apply binary mapping to target variable 
    # Diagnosis = B: 357, M: 212
    breast_cancer_y['is_malignant'] = pd.to_numeric(breast_cancer_y['Diagnosis'].map({'M': 1, 'B': 0}))
    # Drop the original 'Diagnosis' column
    breast_cancer_y.drop(columns=['Diagnosis'], inplace=True)
    
    # Plot the distribution of each explanatory variable (X)
    plot_variable_distribution(breast_cancer_X, "breast_cancer_X")

    # Plot the correlation matrix for explanatory variables (X) and the target variable (y)
    plot_correlation_matrix(breast_cancer_X, breast_cancer_y, "breast_cancer_X_y")

    # Plot the distribution of the target variable (y)
    plot_target_variable_distribution(breast_cancer_y, "breast_cancer_y")

    return breast_cancer_X, breast_cancer_y

def immunotherapy_prep(immunotherapy_X, immunotherapy_y):
    """
    Preprocess the Immunotherapy dataset and generate plots for variable distributions, 
    correlation matrix, and target variable distribution.
    """
    # Apply binary mapping to 'sex' column
    immunotherapy_X['is_female'] = immunotherapy_X['sex'] - 1
    # Drop the original column
    immunotherapy_X.drop(columns=['sex'], inplace=True)

    # Create one-hot encoding for the nominal 'Type' column
    one_hot_encoded_immunotherapy_Type = pd.get_dummies(immunotherapy_X['Type'], prefix='is_type').astype(int)

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    immunotherapy_X = pd.concat([immunotherapy_X, one_hot_encoded_immunotherapy_Type], axis=1)

    # Drop the original nominal 'Type' column
    immunotherapy_X.drop(columns=['Type'], inplace=True)

    # Plot the distribution of each explanatory variable (X)
    plot_variable_distribution(immunotherapy_X, "immunotherapy_X")

    # Plot the correlation matrix for explanatory variables (X) and the target variable (y)
    plot_correlation_matrix(immunotherapy_X, immunotherapy_y, "immunotherapy_X_y")

    # Plot the distribution of the target variable (y)
    plot_target_variable_distribution(immunotherapy_y, "immunotherapy_y")

    return immunotherapy_X, immunotherapy_y

def cryotherapy_prep(cryotherapy_X, cryotherapy_y):
    """
    Preprocess the Cryotherapy dataset and generate plots for variable distributions, 
    correlation matrix, and target variable distribution.
    """
    # Apply binary mapping to 'sex' column
    cryotherapy_X['is_female'] = cryotherapy_X['sex'] - 1
    # Drop the original column
    cryotherapy_X.drop(columns=['sex'], inplace=True)

    # Create one-hot encoding for the nominal 'Type' column
    one_hot_encoded_cryotherapy_Type = pd.get_dummies(cryotherapy_X['Type'], prefix='is_type').astype(int)

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    cryotherapy_X = pd.concat([cryotherapy_X, one_hot_encoded_cryotherapy_Type], axis=1)

    # Drop the original nominal 'Type' column
    cryotherapy_X.drop(columns=['Type'], inplace=True)

    # Plot the distribution of each explanatory variable (X)
    plot_variable_distribution(cryotherapy_X, "cryotherapy_X")

    # Plot the correlation matrix for explanatory variables (X) and the target variable (y)
    plot_correlation_matrix(cryotherapy_X, cryotherapy_y, "cryotherapy_X_y")

    # Plot the distribution of the target variable (y)
    plot_target_variable_distribution(cryotherapy_y, "cryotherapy_y")

    return cryotherapy_X, cryotherapy_y


def save_dataframe_as_csv(dataframe, filename):
    """
    Saves the given DataFrame as a CSV file in the specified directory.
    - None
    """
    # Create the directory if it doesn't exist
    directory = 'Data//Cleaned'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate the file path
    filepath = os.path.join(directory, filename + '.csv')
    
    # Save the DataFrame as a CSV file
    dataframe.to_csv(filepath, index=False)
    print(f"DataFrame saved as '{filepath}'")


# Usage
if __name__ == "__main__":

    # Load all datasets
    datasets = load_datasets()

    # Extract each dataset
    covid_19_X = datasets['covid_19_X']
    covid_19_y = datasets['covid_19_y']
    hepatitis_X = datasets['hepatitis_X']
    hepatitis_y = datasets['hepatitis_y']
    chronic_kidney_disease_X = datasets['chronic_kidney_disease_X']
    chronic_kidney_disease_y = datasets['chronic_kidney_disease_y']
    breast_cancer_X = datasets['breast_cancer_X']
    breast_cancer_y = datasets['breast_cancer_y']
    immunotherapy_X = datasets['immunotherapy_X']
    immunotherapy_y = datasets['immunotherapy_y']
    cryotherapy_X = datasets['cryotherapy_X']
    cryotherapy_y = datasets['cryotherapy_y']

    # Clean each dataset
    covid_19_X, covid_19_y = covid_19_prep(covid_19_X, covid_19_y)
    hepatitis_X_mean_imputted, hepatitis_X_median_imputted, hepatitis_X_missing_dropped, hepatitis_y_missing_dropped = hepatitis_prep(hepatitis_X, hepatitis_y)
    chronic_kidney_disease_X_mean_imputted, chronic_kidney_disease_X_median_imputted, chronic_kidney_disease_X_missing_dropped, chronic_kidney_disease_y_missing_dropped = chronic_kidney_disease_prep(chronic_kidney_disease_X, chronic_kidney_disease_y)
    breast_cancer_X, breast_cancer_y = breast_cancer_prep(breast_cancer_X, breast_cancer_y)
    immunotherapy_X, immunotherapy_y = immunotherapy_prep(immunotherapy_X, immunotherapy_y)
    cryotherapy_X, cryotherapy_y = cryotherapy_prep(cryotherapy_X, cryotherapy_y)

    # Save the COVID-19 X DataFrame as a CSV file
    save_dataframe_as_csv(covid_19_X, "covid_19_X")
    # Save the COVID-19 y DataFrame as a CSV file
    save_dataframe_as_csv(covid_19_y, "covid_19_y")

    # Save the hepatitis X mean-imputed DataFrame as a CSV file
    save_dataframe_as_csv(hepatitis_X_mean_imputted, "hepatitis_X_mean_imputted")
    # Save the hepatitis X median-imputed DataFrame as a CSV file
    save_dataframe_as_csv(hepatitis_X_median_imputted, "hepatitis_X_median_imputted")
    # Save the hepatitis y DataFrame as a CSV file
    save_dataframe_as_csv(hepatitis_y, "hepatitis_y")
    # Save the hepatitis X with missing rows dropped DataFrame as a CSV file
    save_dataframe_as_csv(hepatitis_X_missing_dropped, "hepatitis_X_missing_dropped")
    # Save the hepatitis y with missing rows dropped DataFrame as a CSV file
    save_dataframe_as_csv(hepatitis_y_missing_dropped, "hepatitis_y_missing_dropped")


    # Save the chronic kidney disease X mean-imputed DataFrame as a CSV file
    save_dataframe_as_csv(chronic_kidney_disease_X_mean_imputted, "chronic_kidney_disease_X_mean_imputted")
    # Save the chronic kidney disease X median-imputed DataFrame as a CSV file
    save_dataframe_as_csv(chronic_kidney_disease_X_median_imputted, "chronic_kidney_disease_X_median_imputted")
    # Save the chronic kidney disease y DataFrame as a CSV file
    save_dataframe_as_csv(chronic_kidney_disease_y, "chronic_kidney_disease_y")
    # Save the chronic kidney disease X with missing rows dropped DataFrame as a CSV file
    save_dataframe_as_csv(chronic_kidney_disease_X_missing_dropped, "chronic_kidney_disease_X_missing_dropped")
    # Save the chronic kidney disease y with missing rows dropped DataFrame as a CSV file
    save_dataframe_as_csv(chronic_kidney_disease_y_missing_dropped, "chronic_kidney_disease_y_missing_dropped")


    # Save the breast cancer X DataFrame as a CSV file
    save_dataframe_as_csv(breast_cancer_X, "breast_cancer_X")
    # Save the breast cancer y DataFrame as a CSV file
    save_dataframe_as_csv(breast_cancer_y, "breast_cancer_y")


    # Save the immunotherapy X DataFrame as a CSV file
    save_dataframe_as_csv(immunotherapy_X, "immunotherapy_X")
    # Save the immunotherapy y DataFrame as a CSV file
    save_dataframe_as_csv(immunotherapy_y, "immunotherapy_y")


    # Save the cryotherapy X DataFrame as a CSV file
    save_dataframe_as_csv(cryotherapy_X, "cryotherapy_X")
    # Save the cryotherapy y DataFrame as a CSV file
    save_dataframe_as_csv(cryotherapy_y, "cryotherapy_y")
