import pandas as pd
from loguru import logger

from modern_data_analytics.constants import (
    ACTIVITY_TYPE,
    CONTENT_UPDATE_DATE,
    COUNTRY,
    EC_MAX_CONTRIBUTION,
    EC_SIGNATURE_DATE,
    END_DATE,
    END_OF_PARTICIPATION,
    FRAMEWORK_PROGRAMME,
    FUNDING_SCHEME,
    LEGAL_BASIS,
    MASTER_CALL,
    OBJECTIVE,
    ROLE,
    SME,
    START_DATE,
    STATUS,
    SUB_CALL,
    TITLE,
    TITLE_LEGAL,
    TITLE_TOPIC,
    TOPICS,
    TOTAL_COST,
)
from modern_data_analytics.preprocessing.utils import (
    cast_dtype,
    cast_numeric_with_comma_decimal,
    create_full_project_df,
    legal_summary,
    merge_full_df_with_programme,
    project_feature_engineering,
    project_roles_summary,
    scivoc_summary,
)


def cast_project_df_dtypes(project_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the datatypes of the project dataframe columns to appropriate types.

    Args:
        project_df (pd.DataFrame): Project DataFrame

    Returns:
        pd.DataFrame: DataFrame with casted datatypes
    """
    project_df = cast_dtype(
        project_df, [START_DATE, END_DATE, EC_SIGNATURE_DATE, CONTENT_UPDATE_DATE], target_dtype="datetime"
    )
    project_df = cast_dtype(project_df, [TITLE, OBJECTIVE], target_dtype="string")
    project_df = cast_dtype(
        project_df,
        [STATUS, LEGAL_BASIS, TOPICS, FRAMEWORK_PROGRAMME, FUNDING_SCHEME, MASTER_CALL, SUB_CALL],
        target_dtype="category",
    )
    project_df = cast_numeric_with_comma_decimal(project_df, [TOTAL_COST, EC_MAX_CONTRIBUTION])

    return project_df


def cast_org_df_dtypes(org_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the datatypes of the organisation dataframe columns to appropriate types.

    Args:
        project_df (pd.DataFrame): Organisation DataFrame

    Returns:
        pd.DataFrame: DataFrame with casted datatypes
    """
    org_df = cast_dtype(org_df, [CONTENT_UPDATE_DATE], target_dtype="datetime")
    org_df[TOTAL_COST] = org_df[TOTAL_COST].fillna("0")
    org_df = cast_numeric_with_comma_decimal(org_df, [TOTAL_COST])
    org_df = cast_dtype(org_df, [SME, ROLE, COUNTRY, ACTIVITY_TYPE], target_dtype="category")
    org_df = cast_dtype(org_df, [END_OF_PARTICIPATION], target_dtype="bool")

    return org_df


def cast_topics_df_dtypes(topics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the datatypes of the topics dataframe columns to appropriate types.

    Args:
        project_df (pd.DataFrame): Topics DataFrame

    Returns:
        pd.DataFrame: DataFrame with casted datatypes
    """
    topics_df = cast_dtype(topics_df, [TITLE], target_dtype="string")
    topics_df[TITLE_TOPIC] = topics_df[TITLE]
    topics_df = topics_df.drop(columns=TITLE)

    return topics_df


def cast_legal_df_dtypes(legal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the datatypes of the legal dataframe columns to appropriate types.

    Args:
        project_df (pd.DataFrame): Legal DataFrame

    Returns:
        pd.DataFrame: DataFrame with casted datatypes
    """
    legal_df = cast_dtype(legal_df, [TITLE], target_dtype="string")
    legal_df[TITLE_LEGAL] = legal_df[TITLE]
    legal_df = legal_df.drop(columns=TITLE)

    return legal_df


def preprocess(
    project_df: pd.DataFrame,
    org_df: pd.DataFrame,
    scivoc_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    legal_df: pd.DataFrame,
    programme_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to preprocess the raw dataframes

    Args:
        project_df (pd.DataFrame): Project DataFrame
        org_df (pd.DataFrame): Organisation DataFrame
        scivoc_df (pd.DataFrame): SciVoc DataFrame
        topics_df (pd.DataFrame): Topics DataFrame
        legal_df (pd.DataFrame): Legal DataFrame
        programme_df (pd.DataFrame): Programme DataFrame

    Returns:
        pd.DataFrame: Merged DataFrame with all the processed data
    """
    # Cast datatype
    project_df = cast_project_df_dtypes(project_df)
    org_df = cast_org_df_dtypes(org_df)
    topics_df = cast_topics_df_dtypes(topics_df)
    legal_df = cast_legal_df_dtypes(legal_df)

    # Create summary dataframe
    scivoc_summary_df = scivoc_summary(scivoc_df)
    legal_summary_df = legal_summary(legal_df)
    project_roles_summary_df = project_roles_summary(org_df)

    # Merge dataframes
    full_df = create_full_project_df(
        project_df, project_roles_summary_df, scivoc_summary_df, topics_df, legal_summary_df
    )

    # Feature engineering on full_df
    full_df = project_feature_engineering(full_df)

    # Merge full_df with programme dataframe
    full_merge_df = merge_full_df_with_programme(full_df, programme_df)
    return full_merge_df


def main(
    project_path: str,
    org_path: str,
    scivoc_path: str,
    topics_path: str,
    legal_path: str,
    programme_path: str,
    output_path: str,
) -> None:
    """
    Main function to read input CSVs, process them, and save the output.

    Args:
        project_path (str): Path to project CSV
        org_path (str): Path to organisations CSV
        scivoc_path (str): Path to sciVocTopics CSV
        topics_path (str): Path to topics CSV
        legal_path (str): Path to legal basis CSV
        programme_path (str): Path to framework programme CSV
        output_path (str): Path to save the processed CSV
    """

    project_df = pd.read_csv(project_path)
    org_df = pd.read_csv(org_path)
    scivoc_df = pd.read_csv(scivoc_path)
    topics_df = pd.read_csv(topics_path)
    legal_df = pd.read_csv(legal_path)
    programme_df = pd.read_csv(programme_path)

    processed_df = preprocess(project_df, org_df, scivoc_df, topics_df, legal_df, programme_df)

    processed_df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to: {output_path}")
