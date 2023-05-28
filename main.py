import streamlit as st
import pandas as pd
import lightgbm as lgb
import pickle


header = st.container()


with header:
    st.title('Crop Recommendation App')
    st.text('This app was created to work as a gardening assistant.')
    st.text('With the crop recommendation system, all the guess work is taken out of gardening.')
    st.text('Simply enter readings from your own garden into the User Input Sidebar')
    st.text('and we will suggest the crop best suited to these conditions')


st.sidebar.header('User Input Sidebar')


def user_input_features():
    Nitrogen = st.sidebar.number_input('Input your Nitrogen reading here (0-150)')
    Phosphorus = st.sidebar.number_input('Input your Phosphorus reading here (0-150)')
    Potassium = st.sidebar.number_input('Input your Potassium reading here (0-210)')
    Temperature = st.sidebar.number_input('Input your Temperature reading here (0-50)')
    Humidity = st.sidebar.number_input('Input your Humidity reading here (0-100)')
    PH = st.sidebar.number_input('Input your PH reading here (0-14)')
    Rainfall = st.sidebar.number_input('Input your Rainfall reading here (0-400)')

    data = {'Nitrogen': Nitrogen,
            'Phosphorus': Phosphorus,
            'Potassium': Potassium,
            'Temperature': Temperature,
            'Humidity': Humidity,
            'PH': PH,
            'Rainfall': Rainfall
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
st.write(input_df)


load_clf = pickle.load(open('/Users/alistair/Desktop/STREAM/lgb_model.pkl', 'rb'))
prediction = load_clf.predict(input_df)

st.subheader('Prediction')
st.write(prediction)