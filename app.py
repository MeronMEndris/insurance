from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model('gbr_insurance_charges')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df.iloc[0]['prediction_label']
    return predictions


def run():
    from PIL import Image
    image = Image.open('logo.png')
    image_h = Image.open('hospital.jpg')
    st.image(image)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("One patient", "Multiple patients"))

    st.sidebar.info("This app is created to predict patient hospital charges")
    st.sidebar.success("https://www.pycaret.org")

    st.sidebar.image(image_h)
    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'One patient':
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=100, value=25)
        children = st.selectbox('children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox("Smoker"):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])
        output = ""
        input_dict = {'age': age, 'sex': sex, 'bmi':bmi, 'children': children, 'smoker':smoker, 'region':region}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predit"):
            output = predict(model=model, input_df=input_df)
            output = '$'+str(output)
        st.success("The output is {}".format(output)
                   )
    elif add_selectbox == 'Multiple patients':
        file_upload = st.file_uploader("Upload csv file for predictions" ,type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)
    

if __name__ == '__main__':
    run()