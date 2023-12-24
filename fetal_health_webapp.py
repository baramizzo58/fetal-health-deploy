import streamlit as st
import pickle

gbc_model = pickle.load(open('gbc_model.pkl','rb'))
knn_model = pickle.load(open('knn_model.pkl','rb'))
lr_model = pickle.load(open('logistic_regression_model.pkl','rb'))
rf_model = pickle.load(open('random_forest_model.pkl','rb'))
svm_model = pickle.load(open('svm_model.pkl','rb'))

def classify(num):
    if num<1.5:
        return 'Normal'
    elif num <2.5:
        return 'Suspect'
    else:
        return 'Pathological'

def main():
    st.title("Streamlit Deployment")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Fetal Health Classification</h2>
    </div>
    """
    st.image('dataset-cover.png')
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Gradient Boosting', 'K-Nearest Neighbor', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(f'Machine Learning Model: {option}')
    st.text('This machine learning model was created to predict the health of the fetus carried by the mother. Please select the model you want to use on the left side. Then enter all the parameters below:')
    input1=st.number_input('1. Baseline Fetal Health Rate (FHR): ', min_value=106, max_value=160, placeholder='min 106, max 160')
    st.caption('Note: min 106, max 160')
    input2=st.number_input('2. Number of accelerations per second: ', min_value=0.0000, max_value=0.0190, step=0.0001, format='%.3f', placeholder='min 0, max 0.019')
    st.caption('Note: min 0, max 0.019')
    input3=st.number_input('Number of fetal movements per second: ', min_value=0.0000, max_value=0.4810, step=0.0001, format='%.3f', placeholder='min 0, max 0.481')
    st.caption('Note: min 0, max 0.481')
    input4=st.number_input('Number of uterine contractions per second: ', min_value=0.0000, max_value=0.0150, step=0.0001, format='%.3f', placeholder='min 0, max 0.015')
    st.caption('Note: min 0, max 0.015')
    input5=st.number_input('Number of Light Decelerations (LDs) per second: ', min_value=0.0000, max_value=0.0150, step=0.0001, format='%.3f', placeholder='min 0, max 0.015')
    st.caption('Note: min 0, max 0.015')
    input6=st.number_input('Number of Severe Decelerations (SDs) per second', min_value=0.0000, max_value=0.0010, step=0.0001, format='%.3f', placeholder='min 0, max 0.001')
    st.caption('Note: min 0, max 0.001')
    input7=st.number_input('Number of Prolongued Decelerations (PDs) per second', min_value=0.0000, max_value=0.0050, step=0.0001, format='%.3f', placeholder='min 0, max 0.005')
    st.caption('Note: min 0, max 0.005')
    input8=st.number_input('Percentage of time with abnormal short term variability', min_value=12, max_value=87, placeholder='min 12, max 87')
    st.caption('Note: min 12, max 87')
    input9=st.number_input('Mean value of short term variability', min_value=0.2000, max_value=7.0000, step=0.0001, format='%.3f', placeholder='min 0.2, max 7.0')
    st.caption('Note: min 0.2, max 7.0')
    input10=st.number_input('Percentage of time with abnormal long term variability', min_value=0, max_value=91, placeholder='min 0, max 91')
    st.caption('Note: min 0, max 91')
    input11=st.number_input('Mean value of long term variability', min_value=0.0000, max_value=50.7000, step=0.0001, format='%.3f', placeholder='min 0, max 50.7')
    st.caption('Note: min 0, max 50.7')

    inputs=[[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11]]
    if st.button('Fetal Health Prediction: '):
        if option=='Gradient Boosting':
            st.success(classify(gbc_model.predict(inputs)))
        elif option=='K-Nearest Neighbor':
            st.success(classify(knn_model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(lr_model.predict(inputs)))
        elif option=='Random Forest':
            st.success(classify(rf_model.predict(inputs)))
        else:
           st.success(classify(svm_model.predict(inputs)))


if __name__=='__main__':
    main()
