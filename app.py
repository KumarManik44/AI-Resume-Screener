import streamlit as st
import joblib
import pandas as pd
import numpy as np
from io import StringIO
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')


# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_package = joblib.load('resume_screener_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("Model file 'resume_screener_model.pkl' not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Text preprocessing functions (EXACT same as used in training)
def clean_text(text):
    """Clean resume text - exact same function as training"""
    if pd.isna(text) or text == '' or len(str(text).strip()) == 0:
        return ""

    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)  # Remove phone numbers
    text = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '', text)  # Remove phone numbers
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n+', ' ', text)  # Newlines to spaces
    text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
    text = ' '.join(text.split())  # Clean extra whitespace
    return text.strip()


def advanced_preprocess(text):
    """Advanced preprocessing with tokenization and lemmatization - exact same as training"""
    if not text or text == "":
        return ""

    download_nltk_data()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def preprocess_text(text):
    """Complete preprocessing pipeline - matches training exactly"""
    cleaned_text = clean_text(text)
    processed_text = advanced_preprocess(cleaned_text)
    return processed_text


# Extract text from uploaded file
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.type == "text/plain":
            # Handle .txt files
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
            return text

        elif uploaded_file.type == "application/pdf":
            # For PDF files, you'd need PyPDF2 or similar
            st.warning("PDF upload detected. For this demo, please convert your PDF to text and upload as .txt file.")
            return None

        else:
            st.error("Unsupported file type. Please upload a .txt file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


# Predict resume category
def predict_resume_category(text, model_package):
    # Preprocess using EXACT same pipeline as training
    processed_text = preprocess_text(text)

    # Extract components from model package
    vectorizer = model_package['vectorizer']
    model = model_package['model']
    label_encoder = model_package['label_encoder']

    # Transform text using the vectorizer
    text_vectorized = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    prediction_proba = model.predict_proba(text_vectorized)[0]

    # Get category name using label encoder
    predicted_category = label_encoder.inverse_transform([prediction])[0]
    confidence = prediction_proba[prediction]

    # Get all categories for probability display
    all_categories = label_encoder.classes_
    all_probabilities = dict(zip(all_categories, prediction_proba))

    # Create results dictionary
    results = {
        'predicted_category': predicted_category,
        'confidence': float(confidence),
        'all_probabilities': all_probabilities
    }

    return results


# Main Streamlit app
def main():
    st.set_page_config(
        page_title="AI Resume Screener",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("🤖 AI Resume Screener")
    st.markdown("### Intelligent Resume Category Classification")
    st.markdown("Upload a resume and get instant category prediction with confidence scores!")

    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()

    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")

        # Get model info from the package
        model_info = model_package.get('performance_metrics', {})
        st.info(f"""
        **Algorithm:** {model_package.get('model_name', 'Random Forest')}  
        **Accuracy:** {model_info.get('accuracy', 0.7586):.2%}  
        **Categories:** {model_info.get('total_categories', 24)}  
        **Features:** {model_info.get('feature_count', 5000):,} TF-IDF features
        """)

        st.header("📋 Supported Categories")
        categories = [
            'ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE', 'APPAREL', 'ARTS',
            'AUTOMOBILE', 'AVIATION', 'BANKING', 'BPO', 'BUSINESS-DEVELOPMENT',
            'CHEF', 'CONSTRUCTION', 'CONSULTANT', 'DESIGNER', 'DIGITAL-MEDIA',
            'ENGINEERING', 'FINANCE', 'FITNESS', 'HEALTHCARE', 'HR',
            'INFORMATION-TECHNOLOGY', 'PUBLIC-RELATIONS', 'SALES', 'TEACHER'
        ]
        for category in sorted(categories):
            st.write(f"• {category}")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Upload Resume")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['txt'],
            help="Currently supports .txt files. For PDF files, please convert to text first."
        )

        # Sample text area for demo
        st.subheader("Or paste resume text directly:")
        sample_text = st.text_area(
            "Resume Text",
            height=200,
            placeholder="Paste your resume text here for quick testing..."
        )

        # Quick test samples
        st.subheader("🚀 Quick Test Samples")
        if st.button("Test IT Resume", type="secondary"):
            sample_text = st.session_state.get("sample_it", """
            Software Engineer with 5+ years experience in Python, JavaScript, and machine learning.
            Built scalable web applications using Django and React. Experience with AWS, Docker,
            and database optimization. Strong background in artificial intelligence and data science.
            """)
            st.session_state["sample_it"] = sample_text
            st.rerun()

        if st.button("Test HR Resume", type="secondary"):
            sample_text = st.session_state.get("sample_hr", """
            Human Resources Manager with 8+ years experience in recruitment, employee relations,
            and performance management. Skilled in talent acquisition, HR policies, and team development.
            Experience with HRIS systems and compliance management.
            """)
            st.session_state["sample_hr"] = sample_text
            st.rerun()

        # Predict button
        predict_button = st.button("🔍 Analyze Resume", type="primary")

    with col2:
        st.header("📈 Prediction Results")

        if predict_button:
            # Determine input source
            text_to_analyze = ""

            if uploaded_file is not None:
                text_to_analyze = extract_text_from_file(uploaded_file)
                if text_to_analyze is None:
                    st.stop()
            elif sample_text.strip():
                text_to_analyze = sample_text
            else:
                st.error("Please upload a file or paste resume text.")
                st.stop()

            if len(text_to_analyze.strip()) < 50:
                st.error("Resume text is too short. Please provide a more complete resume.")
                st.stop()

            # Make prediction
            with st.spinner("Analyzing resume..."):
                results = predict_resume_category(text_to_analyze, model_package)

            # Display results
            st.success("Analysis Complete!")

            # Main prediction
            st.subheader("🎯 Predicted Category")
            st.markdown(f"### **{results['predicted_category']}**")
            st.markdown(f"**Confidence:** {results['confidence']:.2%}")

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['confidence'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Top predictions
            st.subheader("📊 Top 5 Predictions")
            sorted_probs = sorted(results['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]

            for i, (category, prob) in enumerate(sorted_probs):
                col_rank, col_cat, col_prob = st.columns([0.5, 2, 1])
                with col_rank:
                    st.write(f"**{i + 1}.**")
                with col_cat:
                    st.write(category)
                with col_prob:
                    st.write(f"{prob:.2%}")

                # Progress bar
                st.progress(prob)

            # Visualization of all categories
            st.subheader("📈 All Category Probabilities")
            prob_df = pd.DataFrame(list(results['all_probabilities'].items()),
                                   columns=['Category', 'Probability'])
            prob_df = prob_df.sort_values('Probability', ascending=True)

            fig_bar = px.bar(
                prob_df.tail(10),
                x='Probability',
                y='Category',
                orientation='h',
                title="Top 10 Category Predictions",
                labels={'Probability': 'Confidence Score', 'Category': 'Job Category'},
                color='Probability',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Resume statistics
            st.subheader("📋 Resume Statistics")
            word_count = len(text_to_analyze.split())
            char_count = len(text_to_analyze)
            processed_text = preprocess_text(text_to_analyze)
            processed_words = len(processed_text.split())

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Word Count", f"{word_count:,}")
            with stat_col2:
                st.metric("Characters", f"{char_count:,}")
            with stat_col3:
                st.metric("Processed Words", f"{processed_words:,}")
            with stat_col4:
                reduction = ((word_count - processed_words) / word_count * 100) if word_count > 0 else 0
                st.metric("Reduction", f"{reduction:.1f}%")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | AI Resume Screener v1.0 | 
        <a href="https://github.com/your-username/resume-screener" target="_blank">View on GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()