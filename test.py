from recommender.matching import get_top_matches
import pandas as pd

project_data = pd.read_csv("data_1/processed/project_merged.csv")

proposal = " war in ukraine"

top_match_ids_scores = get_top_matches(proposal, top_n=10)
ids = [pid for pid, _ in top_match_ids_scores]
scores = {pid: score for pid, score in top_match_ids_scores}

match_df = project_data[project_data["projectID"].isin(ids)].copy()
match_df["similarity"] = match_df["projectID"].map(scores)
match_df.sort_values("similarity", ascending=False, inplace=True)

print(match_df["title"])