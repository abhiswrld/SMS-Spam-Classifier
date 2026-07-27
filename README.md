# SMS Spam Classifier

**Live Demo:** https://abhismsspam.streamlit.app

This is a Natural Language Processing (NLP) machine learning application designed to accurately classify SMS messages as either "Spam" or "Ham" (legitimate). The system analyzes the raw text data using word frequencies and a Multinomial Naive Bayes algorithm, achieving exceptional precision to ensure legitimate messages are never accidentally flagged.

## Key Features
* **100% Precision:** Optimized to completely eliminate False Positives, ensuring no critical messages are sent to the spam folder.
* **Text Preprocessing:** Automatically tokenizes, lowercases, and stems text while stripping punctuation and common English stopwords.
* **Feature Engineering:** Extracts structural insights like character, word, and sentence counts to identify spam patterns.
* **Interactive UI:** A clean web application built entirely in Python using Streamlit.

## The Machine Learning Pipeline
1. **Data Cleaning:** Removed corrupted/unnamed columns, standardized target labels using `LabelEncoder`, and eliminated duplicate records.
2. **Exploratory Data Analysis (EDA):** Visualized feature distributions and generated correlation heatmaps to understand the structural differences between spam and ham.
3. **Text Transformation:** Applied NLTK for word tokenization, removed non-alphanumeric characters, and used a `PorterStemmer` to reduce words to their root forms (e.g., "dancing" and "dancer" become "danc").
4. **Vectorization:** Converted the cleaned text into a mathematical matrix using `TfidfVectorizer`.
5. **Model Training:** Evaluated Gaussian, Multinomial, and Bernoulli Naive Bayes classifiers, ultimately deploying the `MultinomialNB` model due to its flawless precision score.

## Built With
* **Frontend:** [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io/)
* **Machine Learning:** [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
* **NLP & Data Processing:** NLTK, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn

## Running Locally
To run this project on your own machine:
1. Clone the repository
2. Install the required libraries: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app.py`
