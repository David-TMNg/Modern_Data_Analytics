import ast
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ipyleaflet import Icon, Map, Marker
from ipywidgets import HTML
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from modern_data_analytics.recommender import Recommender

# Load project data
project_data = pd.read_csv("data/processed/project_merged.csv")
org_data = pd.read_csv("data/processed/org_unique_detailed.csv")

# Load embeddings to Recommender
with open("models/project_ids.pkl", "rb") as f:
    project_ids = pickle.load(f)
recommender = Recommender()
recommender.load_pretrained_project_embeddings(project_ids, "models/project_embeddings.npy")

# UI
app_ui = ui.page_fluid(
    ui.navset_pill(
        ui.nav_panel(
            "Proposal Match",
            ui.layout_columns(
                ui.card(
                    ui.h2("Welcome to the Horizon Europe Proposal Matcher"),
                    ui.p(
                        "By sharing your proposal, we can help you find similar projects that have already been funded by the European Union."
                        "You can use the tabs above to"
                    ),
                    ui.HTML("""
                        <ul>
                        <li>view individual project details</li>
                        <li>discover potential collaboators</li>
                        <li>explore funding mechanisms</li>
                        </ul>
                        """),
                    ui.input_text_area("proposal", "Enter your research proposal:", rows=6),
                    ui.input_slider("top_n", "Number of results to display:", min=10, max=20, value=10),
                    ui.input_action_button("submit", "Find Matching Projects"),
                ),
                ui.card(ui.output_table("match_summary"))
            )
        ),

        ui.nav_panel(
            "Project Summaries",
            ui.layout_columns(
                ui.card(ui.output_ui("acronym_list"), ui.output_ui("project_detail")),
                ui.accordion(
                    ui.accordion_panel("Organisations Overview", output_widget("map"), ui.output_table("org_summary")),
                    ui.accordion_panel("Funding Overview", ui.output_ui("funding_summary")),
                )
            )
        ),

        ui.nav_panel("Organisation Profile", 
            ui.layout_columns(
                ui.card(ui.output_ui("org_profile_acronym_list"),ui.output_ui("org_profile_org_list"),ui.output_ui("org_profile_summary")),
                ui.card(output_widget("org_profile_map"))
            )
        ),

        ui.nav_panel(
            "Funding Mechanisms",
            ui.card(ui.output_plot("pie_topic"), ui.output_ui("funding_list"), ui.output_ui("funding_detail")),
        ),
        id="tab",
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

        top_match_ids_scores = recommender.get_top_matches(proposal, top_n=input.top_n())
        ids = [pid for pid, _ in top_match_ids_scores]
        scores = {pid: score for pid, score in top_match_ids_scores}

        match_df = project_data[project_data["projectID"].isin(ids)].copy()
        match_df["similarity"] = match_df["projectID"].map(scores)
        match_df.sort_values("similarity", ascending=False, inplace=True)

        matches.set(match_df)

    # helper function to get project organisations from an acronym (used in map rendering)
    def get_project_orgs(acronym):
        df = matches.get()
        orgs = []

        if not acronym:
            return pd.DataFrame()

        row = df[df["acronym"] == acronym].iloc[0]

        role_columns = ["coordinator", "participant", "thirdParty", "associatedPartner"]

        for role in role_columns:
            raw = row.get(role, [])
            if isinstance(raw, str):
                try:
                    ids = ast.literal_eval(raw)
                except Exception:
                    ids = []
            else:
                ids = raw

            if isinstance(ids, list):
                for org_id in ids:
                    orgs.append({"organisationID": org_id, "role": role})
        org_df = pd.DataFrame(orgs)

        # Merge with org_data to get name and location
        result = org_df.merge(org_data, on="organisationID", how="left")
        return result

    # Output the project match summary (Acronym & Title)
    @render.table
    def match_summary():
        df = matches.get()
        if df.empty:
            return pd.DataFrame({"Similar Projects": ["No results yet. Please enter a proposal."]})

        return df[["acronym", "title"]].copy()

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
            ui.a("View full project on CORDIS", href=row["cordis_project_url"], target="_blank"),
        )

    # Output the map
    @render_widget
    def map():
        acronym = input.selected_project()

        if not acronym:
            return Map(center=(50, 10), zoom=4)

        orgs = get_project_orgs(acronym)
        m = Map(center=(50, 10), zoom=4)

        icon = Icon(
            icon_url="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png",
            icon_size=[25, 41],
            icon_anchor=[12, 41],
        )

        for _, row in orgs.iterrows():
            if pd.isna(row["geolocation"]):
                continue

            if row["role"] == "coordinator":
                marker = Marker(
                    icon=icon,
                    location=(row["latitude"], row["longitude"]),
                    title=f"{row['name']} ({row['role']})",
                    draggable=False,
                )

            else:
                marker = Marker(
                    location=(row["latitude"], row["longitude"]),
                    title=f"{row['name']} ({row['role']})",
                    draggable=False,
                )

            m.add(marker)
            marker.popup = HTML(f"<strong>{row['name']}</strong><br>{row['role']}")

        return m

    @render.table
    def org_summary():
        acronym = input.selected_project()
        if not acronym:
            return ui.p("No project selected.")

        df = get_project_orgs(acronym)
        if df.empty:
            return ui.p("No organisations found.")

        return df[["name", "role", "country"]].copy().reset_index(drop=True)

    # Output the project funding summary
    @render.ui
    def funding_summary():
        df = matches.get()
        selected = input.selected_project()
        if not selected:
            return ui.p("Select a project to view details.")

        row = df[df["acronym"] == selected].iloc[0]

        return ui.panel_well(
            ui.p(f"Total Funding: €{row['ecMaxContribution']:,.0f}"),
            ui.p(f"Average Annual Funding per Participant: €{row['avg_annual_funding_per_participant']:,.0f}"),
            ui.p(f"Funding Scheme: {row['funding_id']}"),
            ui.p(f"{row['title_topic']}"),
        )

    @render.ui
    def org_profile_acronym_list():
        df = matches.get()
        if df.empty or "acronym" not in df.columns:
            return ui.p("Submit a proposal first.")
            
        options = df["acronym"].dropna().unique().tolist()
        return ui.input_select("org_selected_acronym", "Select a project acronym:", choices=options)

    @render.ui
    def org_profile_org_list():
        acronym = input.org_selected_acronym()
        if not acronym:
            return ui.p("Select a project acronym first.")

        df = get_project_orgs(acronym)
        if df.empty:
            return ui.p("No organisations found for that project.")

        options = {str(row["organisationID"]): row["name"] for _, row in df.iterrows()}
        return ui.input_select("org_selected_id", "Select an organisation:", choices=options)

    @render.ui
    def org_profile_summary():
        org_id = input.org_selected_id()
        if not org_id:
            return ui.p("Select an organisation.")

        org_id = int(org_id)
        row = org_data[org_data["organisationID"] == org_id]
        if row.empty:
            return ui.p("Organisation not found.")

        row = row.iloc[0]

        content = [
            ui.h4(row["name"]),
            ui.p(f"Projects involved: {int(row['n_projects'])}"),
            ui.p(f"Total funding: €{row['totalCost']:,.0f}")
        ]

        if pd.notna(row["organizationURL"]) and row["organizationURL"].strip():
            content.append(
                ui.a("Organisation Website", href=row["organizationURL"], target="_blank")
            )

        return ui.panel_well(*content)

    @render_widget
    def org_profile_map():
        org_id = input.org_selected_id()
        if not org_id:
            return Map(center=(50, 10), zoom=3)

        org_id = int(org_id)
        row = org_data[org_data["organisationID"] == org_id]
        if row.empty or pd.isna(row.iloc[0]["latitude"]) or pd.isna(row.iloc[0]["longitude"]):
            return Map(center=(50, 10), zoom=3)

        row = row.iloc[0]
        lat, lon = row["latitude"], row["longitude"]

        m = Map(center=(lat, lon), zoom=5)

        marker = Marker(
            location=(lat, lon),
            title=row["name"],
            draggable=False
        )
        marker.popup = HTML(f"<strong>{row['name']}</strong><br>{row['city']}, {row['country']}")
        m.add(marker)

        return m

    # Output the pie chart
    @render.plot
    def pie_topic():
        df = matches.get()
        if df.empty or "title_topic" not in df.columns:
            return

        topic_counts = df["title_topic"].value_counts()
        plt.figure(figsize=(6, 6))
        topic_counts.plot.pie(startangle=90, textprops={"fontsize": 10})
        plt.ylabel("")
        plt.title("Similar projects funded by")
        return plt.gcf()

    # Output the acronym list
    @render.ui
    def funding_list():
        df = matches.get()
        if df.empty or "title_topic" not in df.columns:
            return ui.p("No results yet.")

        options = df["title_topic"].dropna().unique().tolist()

        return ui.input_select("selected_funding", "Select a funding scheme:", choices=options)

    # Output the funding detail
    @render.ui
    def funding_detail():
        df = matches.get()
        selected = input.selected_funding()
        if not selected:
            return ui.p("Select a funding scheme to view details.")

        row = df[df["title_topic"] == selected].iloc[0]

        return ui.panel_well(
            ui.h4(row["title_topic"]),
            ui.HTML(row["topic_objective"]),
            ui.p(""),
            ui.a("View full project on CORDIS", href=row["cordis_funding_url"], target="_blank"),
        )


# App
app = App(app_ui, server)
