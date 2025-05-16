from shiny import App, ui, reactive
import pandas as pd
from recommender.matching import get_top_matches

# Step 1: Load project metadata (merged data)
project_data = pd.read_csv("data_1/processed/project_merged.csv")

# UI
app_ui = ui.page_sidebar(
    title="Horizon Europe Project Recommender",

    sidebar=ui.sidebar(
        ui.input_text_area("proposal", "Enter your research proposal:", rows=6),
        ui.input_action_button("submit", "Find Matching Projects")
    ),

    main=ui.div(
        ui.output_data_frame("match_table")
    )
)

# Server
def server(input, output, session):
    
    # Reactive value to hold the match results
    matches = reactive.Value(pd.DataFrame())

    # When user clicks the button, update matches
    @reactive.effect
    @reactive.event(input.submit)
    def update_matches():
        proposal = input.proposal()
        if not proposal.strip():
            matches.set(pd.DataFrame())  # empty input
            return

        top_match_ids_scores = get_top_matches(proposal, top_n=10)
        ids = [pid for pid, _ in top_match_ids_scores]
        scores = {pid: score for pid, score in top_match_ids_scores}

        match_df = project_data[project_data["projectID"].isin(ids)].copy()
        match_df["similarity"] = match_df["projectID"].map(scores)
        match_df.sort_values("similarity", ascending=False, inplace=True)

        matches.set(match_df[["projectID", "title", "fundingScheme", "totalCost", "similarity"]])

    # Output the table
    @output.data_frame
    def match_table():
        df = matches.get()
        return df if not df.empty else pd.DataFrame({"message": ["Submit a proposal to see matches."]})

# App
app = App(app_ui, server)
