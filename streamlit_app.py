import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

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

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
X = data[features]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

st.title('House Price Prediction App')

st.write("""
This app predicts the price of a house based on specific features. The features used for the prediction are:

- OverallQual: Overall material and finish quality.
- GrLivArea: Above ground living area square feet.
- GarageCars: Size of garage in car capacity.
- TotalBsmtSF: Total square feet of basement area.
- FullBath: Number of full bathrooms.
- YearBuilt: Original construction date.
After inputting the feature values, click the 'Predict' button to see the predicted house price.
""")

overall_qual = st.slider('Overall Quality', min_value=1, max_value=10, value=5)
grliv_area = st.slider('Above Ground Living Area (sq. ft.)', min_value=0, max_value=10000, value=2000)
garage_cars = st.slider('Garage Size (Cars)', min_value=0, max_value=4, value=2)
total_bsmt_sf = st.slider('Total Basement Area (sq. ft.)', min_value=0, max_value=5000, value=1000)
full_bath = st.slider('Number of Full Bathrooms', min_value=0, max_value=4, value=2)
year_built = st.slider('Year Built', min_value=1900, max_value=2023, value=2000)

if st.button('Predict'):
    input_data = np.array([[overall_qual, grliv_area, garage_cars, total_bsmt_sf, full_bath, year_built]])
    prediction = model.predict(input_data)

    st.write('The predicted house price is $', prediction[0])

    display_feature_importance(model, X)
