import pandas as pd
import colorama
from colorama import Fore, Style

def explore_data(data): 
    """
    Explores NaNs, empty spaces and duplicates. Returns a dataframe.
    """
    duplicate_rows = data.duplicated().sum()
    nan_values = data.isna().sum()
    empty_spaces = data.eq(' ').sum()

    if duplicate_rows > 0:
        print(Style.BRIGHT + Fore.RED + f"❗WATCH OUT: There are {duplicate_rows} duplicated rows.")
    else:
        print(Style.BRIGHT + Fore.GREEN + f"✅ There are no duplicated rows!")

    return pd.DataFrame({"nan": nan_values, "empty_spaces": empty_spaces})

def data_analysis(data): 
    """
    Explores data shape, data types and number of unique values. Returns a dataframe.
    """
    data_types = data.dtypes
    nunique_values = data.nunique()

    print(Style.BRIGHT + Fore.BLUE + f"There are {data.shape[0]} rows.")
    print(Style.BRIGHT + Fore.BLUE + f"There are {data.shape[1]} columns.")

    return pd.DataFrame({"data_types": data_types, "nunique_values": nunique_values})

def unique_values(data): 
    """
    Explores columns with fewer than 15 unique values and prints those values in ascending order.
    
    """
    for column in data.columns:
        if data[column].nunique() < 15: # Only shows the values for the columns that have less than 15 unique values
            print(f"The unique values for '{column}' are:")
            print(sorted(map(str, data[column].unique()))) # Convert to string before sorting

def to_snake_case(df):
    """
    Standardizes column names by converting them to lowercase and replacing spaces with underscores.
    """
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    return df

def replace_hyphen(df):
    """
    Standardizes column names by converting them to lowercase and replacing spaces with underscores.
    """
    df.columns = [col.lower().replace("-", "_") for col in df.columns]
    return df

def remove_nans_by_column(df, column_name):
    """
    Removes rows with missing values from the DataFrame.
    """
    return df.dropna(subset=[column_name])

def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()

def fill_nans_mean(df, column_name):
    """
    Fills missing values in a specific column with the mean.
    """
    df[column_name] = df[column_name].fillna(df[column_name].mean())
    return df

def forward_fill_data(df):
    """
    This function applies forward fill (ffill) to fill missing values in the DataFrame.
    """
    return df.ffill()

def convert_to_float(df, column_name):
    """
    Converts a specified column to float format.
    """
    try:
        df[column_name] = df[column_name].astype(float)
    except ValueError as e:
        print(f"Error: Could not convert column '{column_name}' to integer. {e}")
    return df

def convert_to_int(df, column_name):
    """
    Converts a specified column to int format, removes commas.
    """
    try:
        # Remove commas from the column and convert to integers
        df[column_name] = df[column_name].replace({',': ''}, regex=True).astype(int)
    except ValueError as e:
        print(f"Error: Could not convert column '{column_name}' to integer. {e}")
    return df

def remove_outliers(df, column_name):
    """
    Remove outliers
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (df[column_name] > lower_bound) & (df[column_name] < upper_bound)
    return df[mask]