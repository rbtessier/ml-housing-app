import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

#trying to create a custom slider color but it doesnn't like me
def custom_slider_style(color="#be9e44"):  # Use your desired color
    custom_css = f"""
    <style>
        .slider .thumb {{
            background-color: {color} !important;
        }}
        .slider .value {{
            background-color: {color} !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)



def display_feature_importance(model, X):

    #Later I'll make this more global as this is a useful mapping for better code but for now its heara
    description_dict = {
    'GrLivArea': 'Above ground living area square feet.',  
    'GarageCars': 'How many cars the garage fits.',
    'TotalBsmtSF': 'Total square feet of basement area.',
    'BedroomAbvGr': 'Number of above ground bedrooms',
    'FullBath': 'Number of full bathrooms.',
    'YearBuilt': 'Original construction date.'
    }
    
    # Get feature importances-
    importances = model.feature_importances_

    # Create a description column
    description = [description_dict.get(feature, '') for feature in X.columns]
    
    # Associate importances with feature names
    #feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Description' : description})
    feature_importance = pd.DataFrame({'Feature':description, 'Importance': importances})

    
    # Sort features by importance in descending order
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance['Importance'] = feature_importance['Importance'].apply(lambda x: "{:.2f} %".format(x * 100))
    
    # Display the feature importances
    st.dataframe(feature_importance)

data = pd.read_csv("housing.csv")

#features_w_qual = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
features = ['GrLivArea', 'GarageCars', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
X = data[features]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


st.image('streamlit-housing.jpg', caption='A House in Ames, Iowa', use_column_width=True)

st.title('Ames House Price Prediction')

st.write("""
This app predicts the price of a house in **Ames, Iowa** based on specific features of the home. 

After selecting the following features, click the **'Predict'** button at the bottom of the page to see the predicted house price.
""")


custom_slider_style("#be9e44") 
#overall_qual = st.slider('Overall Quality', min_value=1, max_value=10, value=5)
grliv_area = st.slider('Above Ground Living Area (sq. ft.)', min_value=0, max_value=10000, value=2000)
garage_cars = st.slider('Garage Size (Cars)', min_value=0, max_value=4, value=2)
total_bsmt_sf = st.slider('Total Basement Area (sq. ft.)', min_value=0, max_value=5000, value=1000)
bedroom = st.slider('Number of Bedrooms', min_value=0, max_value=8, value=2)
full_bath = st.slider('Number of Full Bathrooms', min_value=0, max_value=4, value=2)
year_built = st.slider('Year the House was Built', min_value=1900, max_value=2023, value=2000)


if st.button('Predict'):
    #input_data_w_features = np.array([[overall_qual, grliv_area, garage_cars, total_bsmt_sf,bedroom, full_bath, year_built]])
    input_data = np.array([[grliv_area, garage_cars, total_bsmt_sf,bedroom, full_bath, year_built]])
    prediction = model.predict(input_data)
    
    
    st.markdown("**The predicted house price is** <span style='color:#be9e44'; font-size:16px;'>**${:,.2f}**</span>".format(prediction[0]), unsafe_allow_html=True)

    st.write('**Feature Importance:** \
    Below is the percentage each feature contributed to the predicted house price')
        
    display_feature_importance(model, X)


