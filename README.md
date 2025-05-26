# Modern Data Analytics Group Assignment
# EU Project Matching System
## Group 25: David O'Grady, David Ng, Liren Xie

This is a Shiny application for matching research proposals with similar EU-funded projects.

The application is currently hosted on AWS at http://13.42.51.136/.

## Features

- Research proposal matching with similar EU projects
- Detailed project information display
- Organization profiles and collaboration networks
- Funding mechanism analysis
- Geographic visualization of project partners

## Prerequisites

- Python 3.10 or higher


## Installation

1. Clone the repository:
```bash
git clone https://github.com/David-TMNg/Modern_Data_Analytics.git
cd Modern-Data-Analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install .
```

For development:
```bash
pip install -e .[dev]
```

## Usage

Run the application locally:
```bash
shiny run app/app.py
```

The application will be available at http://127.0.0.1:8000

## Project Structure
```
Modern_Data_Analytics/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   └── modern_data_analytics/
│       ├── preprocessing/
│       └── recommender/
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml
├── README.md
```

## Data

The application uses processed data from EU-funded projects, including:
- Project details
- Organization information
- Funding mechanisms
- EuroSciVoc topics
