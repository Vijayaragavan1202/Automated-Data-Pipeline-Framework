# Intelligent Data Assistant – Automated Data Preprocessing Framework

This project is a Streamlit-based data preprocessing framework that automates the end-to-end workflow of preparing CSV datasets for machine learning.  
It is designed to help analysts and data scientists clean, encode, and scale data efficiently while maintaining transparency and reproducibility in every step.

---

## Overview

The Intelligent Data Assistant provides an interactive interface to upload datasets and automatically perform preprocessing operations such as:

- Missing value imputation (Mean, Median, or Mode)
- Categorical variable encoding (One-Hot Encoding or Label Encoding)
- Feature scaling (StandardScaler or MinMaxScaler)
- Exploratory data analysis (EDA) through interactive visualizations
- Python code generation for the entire preprocessing workflow
- Python code can be generated in the preset ollama model locally as well.

The application significantly reduces manual effort and promotes consistency in data handling across multiple projects.

---

## Key Features

- **Automated Preprocessing**  
  Automatically identifies numerical and categorical columns and applies the appropriate imputation, encoding, and scaling techniques.

- **Interactive Visualization**  
  Offers histograms, scatter plots, box plots, and correlation heatmaps built with Plotly for better data understanding.

- **Code Reproducibility**  
  Generates a Python script representing all preprocessing operations applied through the interface, enabling reproducible workflows.

- **Customizable Pipeline**  
  Allows users to configure individual steps through the sidebar, making it adaptable to different datasets and project needs.

- **Downloadable Outputs**  
  Cleaned CSV files and generated scripts can be downloaded directly from the interface.

---

## Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Libraries:** pandas, numpy, scikit-learn, plotly, io  
- **Development Tools:** Jupyter, VS Code, Git

---

## Project Structure

intelligent-data-assistant/
│
├── app.py                   # Main Streamlit application
├── requirements.txt         # Project dependencies
├── sample_data/             # Sample CSV datasets
└── README.md                # Project documentation
