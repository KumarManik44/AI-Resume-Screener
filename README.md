# 🤖 AI Resume Screener

An intelligent resume category classification system that automatically categorizes resumes into 24 different job categories using machine learning.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Live Demo

[**Try the App Here**](https://airesumescreener74568.streamlit.app/)

## 📋 Overview

This project implements an end-to-end machine learning pipeline for resume screening and categorization:

- **Data Processing**: Cleaned and preprocessed 2,483 resumes
- **Feature Engineering**: TF-IDF vectorization with 5,000 features  
- **Model Training**: Compared 4 algorithms, Random Forest achieved 75.86% accuracy
- **Model Evaluation**: Comprehensive analysis with confusion matrix and feature importance
- **Deployment**: Complete web application with interactive UI

## 🎯 Supported Categories

The model can classify resumes into 24 categories:

`ACCOUNTANT` `ADVOCATE` `AGRICULTURE` `APPAREL` `ARTS` `AUTOMOBILE` `AVIATION` `BANKING` `BPO` `BUSINESS-DEVELOPMENT` `CHEF` `CONSTRUCTION` `CONSULTANT` `DESIGNER` `DIGITAL-MEDIA` `ENGINEERING` `FINANCE` `FITNESS` `HEALTHCARE` `HR` `INFORMATION-TECHNOLOGY` `PUBLIC-RELATIONS` `SALES` `TEACHER`

## 🏗️ Architecture

```
📁 Project Structure
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── resume_screener_model.pkl       # Trained ML model pipeline
├── resume_predictor.py             # Standalone prediction module
├── model_documentation.json        # Model metadata and info
├── AI Resume Screener.ipynb        # Jupyter Notebook   
└── README.md                       # Project documentation
```

## 🔧 Features

- **📄 File Upload**: Support for .txt resume files
- **✍️ Direct Input**: Paste resume text directly
- **🚀 Quick Test**: Pre-loaded sample resumes for testing
- **📊 Confidence Scoring**: Prediction confidence with visual gauge
- **📈 Top 5 Predictions**: Ranked category predictions
- **📋 Resume Statistics**: Text analysis and preprocessing metrics
- **🎨 Interactive Visualizations**: Plotly charts and graphs

## 🛠️ Installation & Local Setup

### Prerequisites
- Python 3.8+
- pip

### Clone Repository
```bash
git clone https://github.com/your-username/ai-resume-screener.git
cd ai-resume-screener
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | Random Forest |
| **Overall Accuracy** | 75.86% |
| **Categories** | 24 |
| **Features** | 5,000 TF-IDF |
| **Training Time** | 4.77 seconds |

### Category-wise Performance (Top Performers)
- **BUSINESS-DEVELOPMENT**: 100% accuracy
- **INFORMATION-TECHNOLOGY**: 100% accuracy  
- **TEACHER**: 100% accuracy
- **ACCOUNTANT**: 95.8% accuracy
- **DESIGNER**: 95.2% accuracy

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**
   - HTML tag removal
   - URL and email cleaning
   - Text normalization
   - Tokenization and lemmatization

2. **Feature Engineering**
   - TF-IDF vectorization (5,000 features)
   - N-gram range: (1,2)
   - Stop word removal

3. **Model Training**
   - Compared: Naive Bayes, Logistic Regression, Random Forest, SVM
   - Best: Random Forest with 75.86% accuracy
   - Cross-validation and hyperparameter tuning

4. **Model Evaluation**
   - Confusion matrix analysis
   - Feature importance analysis
   - Category-wise performance metrics

## 🚀 Usage

### Web Application
1. Visit the deployed app or run locally
2. Upload a resume (.txt file) or paste text directly
3. Click "Analyze Resume" 
4. View prediction results with confidence scores

### Python API
```python
from resume_predictor import predict_resume_category

# Predict category
result = predict_resume_category("Your resume text here...")
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🔮 Future Enhancements

- [ ] PDF file upload support
- [ ] Batch processing capabilities
- [ ] Resume quality scoring
- [ ] Skills extraction and matching
- [ ] Integration with job boards APIs
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kumar Manik**
- [**LinkedIn**](https://www.linkedin.com/in/kumar2000manik/)

## 🙏 Acknowledgments

- Dataset source and preprocessing techniques
- Streamlit community for excellent documentation
- scikit-learn for machine learning tools

---

⭐ **If you found this project helpful, please give it a star!**
