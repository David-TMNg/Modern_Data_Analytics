from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from modern_data_analytics.constants import (
    ACTIVITY_TYPE,
    ASSOCIATED_PARTNER,
    AVG_ANNUAL_FUNDING_PER_PARTICIPANT,
    AVG_FUNDING_PER_PARTICIPANT,
    CITY,
    CONTENT_UPDATE_DATE,
    COORDINATOR,
    CORDIS_FUNDING_URL,
    CORDIS_PROJECT_URL,
    COUNTRY,
    DURATION_YEARS,
    EC_SIGNATURE_DATE,
    END_DATE,
    EURO_SCIVOC_TITLE,
    FUNDING_ID,
    GEOLOCATION,
    GRANT_DOI,
    ID,
    N_ORGANISATIONS,
    N_PROJECTS,
    N_TITLE_LEGALS,
    NAME,
    NATURE,
    OBJECTIVE,
    ORDER,
    ORGANISATION_ID,
    ORGANIZATION_URL,
    PARTICIPANT,
    PROJECT_ID,
    PROJECTS,
    RCN,
    ROLE,
    SCIVOC_TOPICS,
    SME,
    START_DATE,
    THIRD_PARTY,
    TITLE,
    TITLE_LEGAL,
    TITLE_TOPIC,
    TOPIC_OBJECTIVE,
    TOPICS,
    TOTAL_COST,
)


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

    for col in valid_columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def scivoc_summary(scivoc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes sciVoc topics for each projectID in the SciVoc DataFrame.

    Args:
        scivoc_df (pd.DataFrame): Scivoc DataFrame

    Returns:
        pd.DataFrame: A summary DataFrame of a list of sciVoc topics for each project ID
    """
    required_columns = {PROJECT_ID, EURO_SCIVOC_TITLE}
    if not required_columns.issubset(scivoc_df.columns):
        raise ValueError(f"Scivoc DataFrame must contain the columns: {required_columns}")

    scivoc_summary = scivoc_df.groupby(PROJECT_ID, as_index=False)[EURO_SCIVOC_TITLE].agg(list)

    scivoc_summary = scivoc_summary.rename(columns={EURO_SCIVOC_TITLE: SCIVOC_TOPICS})
    scivoc_summary[SCIVOC_TOPICS] = scivoc_summary[SCIVOC_TOPICS].apply(lambda x: x if isinstance(x, list) else [])
    return scivoc_summary


def legal_summary(legal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes legal titles for each projectID in the Legal Basis DataFrame.

    Args:
        legal_df (pd.DataFrame): Legal Basis DataFrame

    Returns:
        pd.DataFrame: A summary DataFrame of a list of legal titles and number
        of legal titles for each project ID
    """

    required_columns = {PROJECT_ID, TITLE_LEGAL}
    if not required_columns.issubset(legal_df.columns):
        raise ValueError(f"Legal DataFrame must contain the columns: {required_columns}")

    legal_summary = (
        legal_df.dropna(subset=[TITLE_LEGAL])
        .groupby(PROJECT_ID, as_index=False)[TITLE_LEGAL]
        .apply(lambda x: sorted(set(x)))
    )

    legal_summary[N_TITLE_LEGALS] = legal_summary[TITLE_LEGAL].str.len()
    return legal_summary


def org_summary(org_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes project information for each organisation.

    Args:
        org_df (pd.DataFrame): Organisation DataFrame

    Returns:
        pd.DataFrame: A summary DataFrame with one row per organisation, containing organisation
        information and aggregated project data
    """
    # Project details for each organisation
    org_projects = (
        org_df.groupby(ORGANISATION_ID)[[PROJECT_ID, ORDER, ROLE, TOTAL_COST]]
        .apply(lambda df: df.to_dict("records"))
        .reset_index(name=PROJECTS)
    )

    # Organisation information
    org_info_cols = [
        ORGANISATION_ID,
        NAME,
        SME,
        ACTIVITY_TYPE,
        COUNTRY,
        CITY,
        GEOLOCATION,
        ORGANIZATION_URL,
    ]
    org_info = org_df.drop_duplicates(subset=ORGANISATION_ID)[org_info_cols]

    # Merge project data into org info
    org_summary = org_info.merge(org_projects, on=ORGANISATION_ID, how="left")

    # Add number of projects and total cost
    org_summary[N_PROJECTS] = org_summary[PROJECTS].apply(len)
    org_summary[TOTAL_COST] = org_summary[PROJECTS].apply(
        lambda projects: sum(proj.get(TOTAL_COST, 0) or 0 for proj in projects)
    )

    return org_summary


def project_roles_summary(org_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes roles for each project.

    Args:
        org_df (pd.DataFrame): Organisation DataFrame

    Returns:
        pd.DataFrame: A summary dataframe with listing the roles for each project ID
    """
    # Initialize role categories
    roles = {COORDINATOR, PARTICIPANT, THIRD_PARTY, ASSOCIATED_PARTNER}

    # Initialize role lists per project
    project_roles: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {role: [] for role in roles})

    # Populate roles
    for _, row in org_df.iterrows():
        pid = row[PROJECT_ID]
        role = str(row[ROLE]).strip() if pd.notna(row[ROLE]) else None
        org_id = row[ORGANISATION_ID]

        if role in roles:
            project_roles[pid][role].append(org_id)

    # Summary dict of each project ID
    project_records = [
        {
            PROJECT_ID: pid,
            COORDINATOR: roles[COORDINATOR],
            PARTICIPANT: roles[PARTICIPANT],
            THIRD_PARTY: roles[THIRD_PARTY],
            ASSOCIATED_PARTNER: roles[ASSOCIATED_PARTNER],
            N_ORGANISATIONS: sum(len(orgs) for orgs in roles.values()),
        }
        for pid, roles in project_roles.items()
    ]

    return pd.DataFrame(project_records)


def create_full_project_df(
    project_df: pd.DataFrame,
    project_summary: pd.DataFrame,
    scivoc_summary: pd.DataFrame,
    topics_df: pd.DataFrame,
    legal_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create merged project DataFrame with the base project data and summaries

    Args
        project_df (pd.DataFrame): Base project DataFrame
        project_summary (pd.DataFrame): Result of project_summary()
        scivoc_summary (pd.DataFrame): Result of scivoc_summary()
        topics_df (pd.DataFrame): Raw topic data
        legal_summary (pd.DataFrame): Result of legal_summary()

    Returns:
        pd.DataFrame: full dataset of projects and summaries
    """
    project_df = project_df.rename(columns={ID: PROJECT_ID})

    merged = (
        project_df.merge(project_summary, on=PROJECT_ID, how="left")
        .merge(scivoc_summary, on=PROJECT_ID, how="left")
        .merge(topics_df[[PROJECT_ID, TITLE_TOPIC]], on=PROJECT_ID, how="left")
        .merge(legal_summary, on=PROJECT_ID, how="left")
    )

    columns_to_drop = [
        EC_SIGNATURE_DATE,
        NATURE,
        CONTENT_UPDATE_DATE,
        RCN,
        GRANT_DOI,
    ]
    merged = merged.drop(columns=columns_to_drop, errors="ignore")

    return merged


def project_feature_engineering(full_project_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes additional features on project duration in years, average funding per participant,
    and average annual funding per partcipant on the full project DataFrame

    Args:
        full_project_df (pd.DataFrame): Result of create_full_project_df()

    Returns:
        pd.DataFrame: full project DataFrame with new features
    """
    # Error handling to avoid zero division error
    valid_orgs = full_project_df[N_ORGANISATIONS].replace(0, np.nan)
    # Average funding per participant
    full_project_df[AVG_FUNDING_PER_PARTICIPANT] = (full_project_df[TOTAL_COST] / valid_orgs).round(2)

    # Calculate duration in years
    full_project_df[DURATION_YEARS] = (full_project_df[END_DATE] - full_project_df[START_DATE]).dt.days / 365.25

    # Error handling to avoid zero division error
    valid_duration = full_project_df[DURATION_YEARS].replace(0, np.nan)

    # Average annual funding per participant
    full_project_df[AVG_ANNUAL_FUNDING_PER_PARTICIPANT] = (
        full_project_df[TOTAL_COST] / (valid_orgs * valid_duration)
    ).round(2)

    project_url = "https://cordis.europa.eu/project/id/"
    full_project_df[CORDIS_PROJECT_URL] = project_url + full_project_df[PROJECT_ID].astype(str)

    funding_url = "https://cordis.europa.eu/programme/id/HORIZON_"
    full_project_df[CORDIS_FUNDING_URL] = funding_url + full_project_df[TOPICS].astype(str)

    full_project_df[FUNDING_ID] = "HORIZON_" + full_project_df[TOPICS].astype(str)

    return full_project_df


def merge_full_df_with_programme(full_df: pd.DataFrame, programme_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge full project DataFrame withe programme DataFrame

    Args:
        full_df (pd.DataFrame): Result of project_feature_engineering()

    Return:
        pd.DataFrame: full project DataFrame with programme DataFrame
    """
    programme_df = programme_df[[ID, OBJECTIVE]].rename(columns={ID: FUNDING_ID, OBJECTIVE: TOPIC_OBJECTIVE})
    full_merged_df = full_df.merge(programme_df, on=FUNDING_ID, how="left")

    return full_merged_df


def format_input_text(full_df: pd.DataFrame) -> pd.Series:
    """
    Format input text for recommender training by combining title, objective, and SciVoc topics
    Args:
        full_df (pd.DataFrame): Full preprocessed DataFrame
    Returns:
        pd.Series: Series containing formatted input text for each project
    """

    input_text = (
        full_df[TITLE].fillna("")
        + " "
        + full_df[OBJECTIVE].fillna("")
        + " "
        + full_df[SCIVOC_TOPICS].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    )
    return input_text
