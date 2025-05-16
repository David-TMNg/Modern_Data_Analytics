from shiny import App, ui, reactive, render
import pandas as pd
from recommender.matching import get_top_matches
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load project metadata (merged data)
project_data = pd.read_csv("data_1/processed/project_merged.csv")

# UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_text_area("proposal", "Enter your research proposal:", rows=6),
        ui.input_slider("top_n", "Number of results to display:", min=10, max=50, value=10),
        ui.input_action_button("submit", "Find Matching Projects")
    ),

    ui.layout_columns(
        ui.card(
            ui.output_ui("acronym_list")
        ),

        ui.card(
            ui.card(ui.output_plot("pie_topic")),
            ui.card(ui.output_plot("boxplot_funding"))
        ),

        col_widths=(4, 8)
    ),

    ui.output_ui("project_detail"),

    title="Horizon Europe Project Recommender",
    
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

        top_match_ids_scores = get_top_matches(proposal, top_n=input.top_n())
        ids = [pid for pid, _ in top_match_ids_scores]
        scores = {pid: score for pid, score in top_match_ids_scores}

        match_df = project_data[project_data["projectID"].isin(ids)].copy()
        match_df["similarity"] = match_df["projectID"].map(scores)
        match_df.sort_values("similarity", ascending=False, inplace=True)

        matches.set(match_df)

    # Output the boxplot
    @render.plot
    def boxplot_funding():
        df = matches.get()
        if df.empty:
            return

        plt.figure(figsize=(6, 2))
        sns.boxplot(x=df["ecMaxContribution"]/1e6)
        plt.title("Project Funding")
        plt.xlabel("Funding in millions€")
        return plt.gcf()

    # Output the pie chart
    @render.plot
    def pie_topic():
        df = matches.get()
        if df.empty or "title_topic" not in df.columns:
            return

        topic_counts = df["title_topic"].value_counts()
        plt.figure(figsize=(6, 6))
        topic_counts.plot.pie(startangle=90,textprops={'fontsize': 10})
        plt.ylabel("")
        plt.title("Grants awarded by")
        return plt.gcf()

    # Output the acronym list
    @render.ui
    def acronym_list():
        df = matches.get()
        if df.empty or "acronym" not in df.columns:
            return ui.p("No results yet.")

        options = df["acronym"].dropna().unique().tolist()
        return ui.input_select("selected_project", "Select a project acronym:", choices=options)

    # Output the project detail
    @render.ui
    def project_detail():
        df = matches.get()
        selected = input.selected_project()
        if not selected:
            return ui.p("Select a project to view details.")

        row = df[df["acronym"] == selected].iloc[0]

        return ui.panel_well(
            ui.h4(row["title"]),
            ui.p(f"Objective: {row['objective']}"),
            ui.p(f"Funding Scheme: {row['fundingScheme']}"),
            ui.p(f"Total Cost: €{row['totalCost']:,.0f}"),
            ui.p(f"Start: {row['startDate']} | End: {row['endDate']}"),
            ui.p(f"Participants: {row['n_organisations']}")
        )


# App
app = App(app_ui, server)
