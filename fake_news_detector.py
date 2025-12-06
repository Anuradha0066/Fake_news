import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# --------------------------------------
# LOAD AND TRAIN MODEL
# --------------------------------------
@st.cache_data
def load_and_train_model():
    df_true = pd.read_csv("True.csv")
    df_fake = pd.read_csv("Fake.csv")

    df_fake["class"] = 0
    df_true["class"] = 1

    df = pd.concat([df_fake, df_true], axis=0)
    df = df.drop(["title", "subject", "date"], axis=1)

    df.drop_duplicates(inplace=True)
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["class"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(xv_train, y_train)

    accuracy = model.score(xv_test, y_test)

    return model, vectorizer, accuracy


# --------------------------------------
# STREAMLIT UI DESIGN
# --------------------------------------
def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

    # Initialize session state variable
    if "news_input" not in st.session_state:
        st.session_state["news_input"] = ""

    st.title("📰 Fake News Detection using Machine Learning")
    st.write("Enter any news text below and check whether it's **Real** or **Fake** 👇")

    st.sidebar.title("⚙️ Model Information")

    with st.spinner("Training AI Model... Please wait"):
        model, vectorizer, accuracy = load_and_train_model()

    st.sidebar.success(f"Accuracy: {accuracy * 100:.2f}%")

    st.subheader("🧪 Try Sample Examples")

    col1, col2, col3 = st.columns(3)

    sample_real = "Government announces new policies to support rural employment and education."
    sample_fake = "Scientists confirm the earth will flip upside down next week due to moon vibration."
    sample_real2 = "WHO confirms decline in cases due to improved vaccination programs."

    with col1:
        if st.button("Example Real 1"):
            st.session_state["news_input"] = sample_real

    with col2:
        if st.button("Example Fake"):
            st.session_state["news_input"] = sample_fake

    with col3:
        if st.button("Example Real 2"):
            st.session_state["news_input"] = sample_real2

    user_text = st.text_area("✍️ Enter News Text Here:", value=st.session_state["news_input"])

    if st.button("Predict"):
        if user_text.strip() == "":
            st.warning("⚠️ Please enter some news text!")
        else:
            cleaned = clean_text(user_text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)

            if prediction[0] == 1:
                st.success("🟢 REAL NEWS — Trustworthy Source")
            else:
                st.error("🔴 FAKE NEWS — Unverified or Misleading Content")

    st.markdown("---")
    st.caption("Have a Tea with real news ☕😊")

if __name__ == "__main__":
    main()
