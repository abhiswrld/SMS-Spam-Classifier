import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK datasets for Streamlit Cloud
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    new_text = []
    for i in text:
        if i.isalnum():
            new_text.append(i)

    text = new_text[:]
    new_text.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            new_text.append(i)

    text = new_text[:]
    new_text.clear()

    for i in text:
        new_text.append(ps.stem(i))

    return " ".join(new_text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")
sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = transform_text(sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")