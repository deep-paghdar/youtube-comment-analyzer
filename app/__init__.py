from flask import Flask
import joblib

# Initialize Flask app
app = Flask(__name__)

#Load model at the package level
model = joblib.load("app/model.pkl")
from app import routes