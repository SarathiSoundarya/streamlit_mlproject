import streamlit as st
import pickle
import pandas as pd
from io import StringIO

# Load the trained classifier from pickle
def load_classifier():
    with open("notebook/classifier.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
    return classifier

classifier = load_classifier()

# Function to handle user input and make prediction
def predict_note_authentication(classifier, variance, skewness, curtosis, entropy):
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return prediction[0]

# Function to handle file upload and make predictions
def handle_file_upload(classifier, uploaded_file):
    contents = uploaded_file.getvalue()  # Get the byte data of the file
    str_content = contents.decode('utf-8')  # Convert bytes to string
    data = StringIO(str_content)  # Convert string to StringIO for pandas to read
    df_test = pd.read_csv(data)  # Read the CSV into a pandas DataFrame

    # Make predictions for the uploaded file
    predictions = classifier.predict(df_test)
    
    # Add predictions to the DataFrame
    df_test['Prediction'] = predictions
    return df_test

# Streamlit App Layout
def main():
    st.title("Bank Note Auth Prediction")

    # Sidebar for user inputs
    st.sidebar.header("Input Features")
    input_method = st.sidebar.radio("Select Input Method", ("Enter Features Manually", "Upload CSV File"))

    if input_method == "Enter Features Manually":
        variance = st.sidebar.number_input("Variance", value=0.0)
        skewness = st.sidebar.number_input("Skewness", value=0.0)
        curtosis = st.sidebar.number_input("Curtosis", value=0.0)
        entropy = st.sidebar.number_input("Entropy", value=0.0)

        if st.sidebar.button('Predict'):
            result = predict_note_authentication(classifier, variance, skewness, curtosis, entropy)
            st.write(f"The predicted value is: {result}")

    elif input_method == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                if st.sidebar.button('Predict'):
                    df_test = handle_file_upload(classifier, uploaded_file)
                    st.write("Predictions for the uploaded file:")
                    st.write(df_test)
            else:
                st.sidebar.error("The uploaded file is not a CSV file. Please upload a valid CSV file.")

    
if __name__ == "__main__":
    main()
