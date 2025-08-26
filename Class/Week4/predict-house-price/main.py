import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def main():

	# 1. Create a synthetic dataset (replace with your actual data)
	data = {
	    'SqFt': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200],
	    'Bedrooms': [2, 3, 3, 4, 4, 4, 5, 5, 5, 6],
	    'Price': [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000]
	}
	df = pd.DataFrame(data)

	# 2. Define features (X) and target (y)
	X = df[['SqFt', 'Bedrooms']]
	y = df['Price']

	# 3. Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 4. Choose and train the model
	model = LinearRegression()
	model.fit(X_train, y_train)

	# 5. Make predictions
	y_pred = model.predict(X_test)

	# 6. Evaluate the model
	mse = mean_squared_error(y_test, y_pred)
	print(f"Mean Squared Error: {mse:.2f}")

	# 7. Make a new prediction (example)
	new_house_data = pd.DataFrame([[1600, 3]], columns=['SqFt', 'Bedrooms'])
	predicted_price = model.predict(new_house_data)
	print(f"Predicted price for a new house: ${predicted_price[0]:.2f}")

if __name__ == "__main__":
    main()
