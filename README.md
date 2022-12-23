# Airbnb Price prediction app using Flask and XGBoost

- This is a web application developed using the Flask framework that tries to predict the price of an Airbnb listings in Santiago using certain characteristics:

  - Location in (latitude, longitude) coordinates
  - Number of guests
  - Number of bedrooms
  - Number of beds
  - 30 day availability
  
  ![image](https://user-images.githubusercontent.com/69170636/208762914-f4ee52f4-8eeb-4eea-bf4f-c58facb19abf.png)

- The XGBoost model is trained using Airbnb listings from [Inside Airbnb](http://insideairbnb.com). Before training the model the data was cleaned, removing rows with null values, certain outlying values and columns that were not relevant to the problem. The trained model is then saved to a Pickle file, and a REST API was created to make POST requests on the model with the features specified earlier. 

- The project was deployed on and AWS instance (now deprecated), but can be run on localhost using port 5000, by running `python app.py`.

