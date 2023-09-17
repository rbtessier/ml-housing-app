import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

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
    # Get feature importances-
    importances = model.feature_importances_
    # Associate importances with feature names
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

    # Sort features by importance in descending order
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # Display the feature importances
    st.dataframe(feature_importance)

data = pd.read_csv("housing.csv")

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
X = data[features]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

st.title('Ames House Price Prediction')

st.write("""
This app predicts the price of a house in **Ames, Iowa** based on specific attribures of the home. 
         
The attributes used are as follows:

- Overall Quality[OverallQual]: Overall material and finish quality as assessed on appraisal.
- Above Ground Living Area (sq. ft.)[GrLivArea]: Above ground living area square feet.
- Garage Size (Cars) [GarageCars]: Size of garage in car capacity.
- Total Basement Area (sq. ft.) [TotalBsmtSF]: Total square feet of basement area.
- Number of Bedrooms [BedroomAbvGr]: Number of above ground bedrooms
- Number of Full Bathrooms [FullBath]: Number of full bathrooms.
- Year the House was Built [YearBuilt]: Original construction date.
         
After selecting the attributes value of the house you want to know the price of, click the **'Predict'** button at the bottom of the page to see the predicted house price.
""")

custom_slider_style("#be9e44") 
overall_qual = st.slider('Overall Quality', min_value=1, max_value=10, value=5)
grliv_area = st.slider('Above Ground Living Area (sq. ft.)', min_value=0, max_value=10000, value=2000)
garage_cars = st.slider('Garage Size (Cars)', min_value=0, max_value=4, value=2)
total_bsmt_sf = st.slider('Total Basement Area (sq. ft.)', min_value=0, max_value=5000, value=1000)
bedroom = st.slider('Number of Bedrooms', min_value=0, max_value=8, value=2)
full_bath = st.slider('Number of Full Bathrooms', min_value=0, max_value=4, value=2)
year_built = st.slider('Year the House was Built', min_value=1900, max_value=2023, value=2000)


if st.button('Predict'):
    input_data = np.array([[overall_qual, grliv_area, garage_cars, total_bsmt_sf,bedroom, full_bath, year_built]])
    prediction = model.predict(input_data)

    st.write('The predicted house price is $', prediction[0])

    display_feature_importance(model, X)



