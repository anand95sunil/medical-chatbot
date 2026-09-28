import pickle
import random
import streamlit as st


# Load trained model
with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
vectorizer = model_data["vectorizer"]
responses = model_data["responses"]


# Page settings
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥"
)

st.title("🏥 Medical Chatbot")
st.write("Ask me a basic health-related question.")


# User input
user_input = st.text_input("You:")


if user_input:

    # Convert user text into TF-IDF features
    user_vector = vectorizer.transform([user_input])

    # Predict intent
    predicted_intent = model.predict(user_vector)[0]

    # Generate response
    response = random.choice(responses[predicted_intent])

    # Display response
    st.write("🤖 **Chatbot:**")
    st.write(response)