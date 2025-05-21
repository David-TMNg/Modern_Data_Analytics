import pandas as pd


def cast_dtype(df: pd.DataFrame, columns: list, target_dtype: str) -> pd.DataFrame:
    """
    Cast multiple columns to the same data type

    Args:
        df (pd.DataFrame): input DataFrame
        columns (list): list of column names to cast
        target_dtype (str): The target data type: "bool", "string", "datetime" or "category"

    Returns:
        Modified DataFrame
    """
    valid_columns = [col for col in columns if col in df.columns]

    if valid_columns != columns:
        invalid_columns = [col for col in columns if col in df.columns]
        raise ValueError(f"Columns {invalid_columns} do not exist")

    if target_dtype not in ["bool", "string", "datetime", "category"]:
        raise ValueError("Invalid datatype")

    if target_dtype == "datetime":
        df[valid_columns] = df[valid_columns].apply(pd.to_datetime, errors="coerce")
    else:
        df[valid_columns] = df[valid_columns].astype(target_dtype)

    return df


def cast_numeric_with_comma_decimal(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Cast columns containing numeric values with commas as decimal separators to
    numeric types.

    Args:
        df (pd.DataFrame): input DataFrame
        columns (list): list of column names to cast

    Returns:
        Modified DataFrame
    """

    valid_columns = [col for col in columns if col in df.columns]

    if valid_columns != columns:
        invalid_columns = [col for col in columns if col in df.columns]
        raise ValueError(f"Columns {invalid_columns} do not exist")

    df[valid_columns] = df[valid_columns].astype(str)
    df[valid_columns] = df[valid_columns].str.replace(",", ".")
    df[valid_columns] = pd.to_numeric(df[valid_columns], errors="coerce")

    return df


def impute_missing_with_mean():
    pass


def scivoc_summary():
    pass


def legal_summary():
    pass


def drop_columns():
    pass
