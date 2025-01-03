# Youtube Comment Analyzer Project

This project provides an end-to-end solution for analyzing YouTube comments using a machine learning model. It consists of a Google Chrome extension for collecting comments and displaying analysis results, a Flask backend serving the ML model, and deployment on AWS using Docker. MLflow is used for experiment tracking and model management.

## Features
    Analyze YouTube comments for sentiment, toxicity, or topics.
    Chrome extension for seamless comment collection and visualization.
    Flask API serving a machine learning model with real-time responses.
    Dockerized application deployed on AWS.
    Experiment tracking and model management using MLflow

## Tech Stack
- Python Version 3.10+
- Flask
- MlFlow
- HTML/CSS
- Docker


## Setup Guide
### 1. Local Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/youtube-comment-analyzer.git
   cd youtube-comment-analyzer
   ```
2. Create and activate a virtual environment:
   ```bash
   conda create --name venv python=3.10
   conda activate venv
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run a service locally:
   ```bash
   python run.py
   ```