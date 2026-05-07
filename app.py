import gradio as gr
import pandas as pd
import joblib
import emoji
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PREPROCESSING FUNCTION (Must match Notebook) ---
# This matches the logic from your 'Aiproject 2.ipynb'
def preprocess_text(text_series):
    # Ensure string format
    text = text_series.astype(str)
    text = text.str.lower()
    # Regex cleaning (same as notebook)
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace(r'\b\d+\b', 'NUMBER', regex=True)
    text = text.str.replace(r'(http|https)://\S+|www\.\S+', 'URL', regex=True)
    # Emoji handling (Crucial!)
    text = text.apply(lambda x: emoji.demojize(x))
    return text

# --- 2. LOAD THE SAVED PIPELINE ---
model = None
vectorizer = None
status = "Initializing..."

try:
    if os.path.exists('best_model2.pkl'):
        model = joblib.load('best_model2.pkl')
    
    if os.path.exists('tfidf_vectorizer.pkl'):
        vectorizer = joblib.load('tfidf_vectorizer.pkl')

    if model and vectorizer:
        status = "✅ System Ready: ML Pipeline Loaded."
    else:
        status = "⚠️ ERROR: Model files missing. Please upload 'best_model2.pkl' and 'tfidf_vectorizer.pkl'."
        
except Exception as e:
    status = f"❌ CRITICAL LOAD ERROR: {str(e)}"

# --- 3. CLASSIFICATION FUNCTION ---
def classify_file(file_obj):
    # Safety Check
    if model is None or vectorizer is None:
        return None, None, f"System Error: {status}"
    
    if file_obj is None:
        return None, None, "Please upload a CSV file."

    try:
        # Read File
        df = pd.read_csv(file_obj.name)
        
        # Find Text Column
        text_col = None
        # Prioritize columns named 'Reviews' or 'text'
        for col in df.columns:
            if col.lower() in ['reviews', 'text', 'review']:
                text_col = col
                break
        
        # Fallback to first text column
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
                    
        if text_col is None:
            return None, None, "Error: Could not find a text column in CSV."

        # Apply Pipeline: Preprocess -> Vectorize -> Predict
        clean_text = preprocess_text(df[text_col])
        features = vectorizer.transform(clean_text)
        predictions = model.predict(features)
        
        # Map Labels (0=Positive, 1=Negative, 2=Neutral)
        label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
        df['Predicted_Sentiment'] = [label_map.get(p, "Unknown") for p in predictions]

        # Generate Chart
        plt.figure(figsize=(8, 4))
        sns.countplot(x='Predicted_Sentiment', data=df, palette='viridis')
        plt.title('Sentiment Analysis Results')
        plt.tight_layout()
        plot_path = "results.png"
        plt.savefig(plot_path)
        plt.close()

        return df, plot_path, "✅ Analysis Complete!"

    except Exception as e:
        return None, None, f"Runtime Error: {e}"

# --- 4. LAUNCH APP ---
iface = gr.Interface(
    fn=classify_file,
    inputs=gr.File(label="Upload CSV (e.g. Nouman1.csv)"),
    outputs=[
        gr.Dataframe(label="Results"),
        gr.Image(label="Chart"),
        gr.Textbox(label="System Status")
    ],
    title="ML Pipeline Project",
    description=f"Current Status: {status}"
)

iface.launch()