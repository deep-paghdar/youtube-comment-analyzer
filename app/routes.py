from flask import request, render_template, redirect, url_for
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np
from . import app, model
from .models.utils import preprocess_text, is_english_sentence, remove_emojis

# from flask import render_template, request, redirect, url_for
# from app import app
# YouTube API setup
API_KEY = 'AIzaSyCnFnJYsL5YtHxCbiMiqRiH-sjn2mHIw4M'  # Replace with your YouTube API key
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_youtube_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText")
    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    return comments

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_url = request.form["video_url"]
        return redirect(url_for("results", video_url=video_url))
    return render_template("index.html")

@app.route("/results", methods=["GET"])
def results():
    video_url = request.args.get("video_url")
    video_id = video_url.split("v=")[-1]

    comments = get_youtube_comments(video_id)

    #preprocess the data
    df = pd.DataFrame({'text':comments})
    df_no_emojis = df.applymap(lambda x: remove_emojis(str(x)))
    df_english = df_no_emojis[df_no_emojis["text"].apply(is_english_sentence)]
    df_english['preprocess_text'] = df_english['text'].apply(preprocess_text)

    #remove null values
    df_english.dropna(inplace=True)

    # Predict labels for each comment
    predictions_prob = model.predict_proba(df_english['preprocess_text'])  # Assuming model.predict() takes a list of comments
    predictions_label = np.argmax(predictions_prob, axis=1)
    label_prob = np.max(predictions_prob, axis=1)

    df_english['predict_prob'] = label_prob
    df_english['prediction_label'] = predictions_label

    df_english['prediction_category'] = df_english['prediction_label'].map({0:'neutral', 1:'positive', 2:'negative'})

    # Store predictions and categories
    categories = ["neutral", "positive", "negative"]  # Modify based on your model's categories
    results = {category: 0 for category in categories}

    for prediction in df_english['prediction_category']:
        results[prediction] += 1

    # Create pie chart
    df = pd.DataFrame(list(results.items()), columns=["Category", "Count"])
    plt.figure(figsize=(6, 6))
    plt.pie(df["Count"], labels=df["Category"], autopct='%1.1f%%', startangle=90)
    plt.title("Comment Category Distribution")

    # Save pie chart to a byte object for rendering in HTML
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format="png")
    img_stream.seek(0)
    pie_chart_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

    df_top = df_english[df_english['prediction_label']==0].sort_values('predict_prob', ascending=False).head(10)
    top_comments = [tuple(x) for x in df_top[['text', 'predict_prob', 'prediction_category']].values]

    return render_template("results.html", pie_chart=pie_chart_base64, predictions=results, top_comments=top_comments)

# from flask import render_template, request, redirect, url_for
# from app import app

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         video_url = request.form["video_url"]
#         return redirect(url_for("results", video_url=video_url))
#     return render_template("index.html")

# @app.route("/results", methods=["GET"])
# def results():
#     video_url = request.args.get("video_url")
#     return f"Received video URL: {video_url}"
