import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error

data = {
    "Size_sqft": [800, 900, 1000, 1100, 1200, 1300, 1500, 1700, 2000, 2300],
    "Bedrooms":  [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    "Bathrooms": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    "Price":     [40, 50, 55, 60, 70, 80, 95, 110, 140, 170]
}

df = pd.DataFrame(data)
X = df.drop("Price",axis=1)
y = df["Price"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=42)
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr_model = LinearRegression()
lr_model.fit(x_train_scaled,y_train)

y_test_pred = lr_model.predict(x_test_scaled)

lr_r2 = r2_score(y_test,y_test_pred)
lr_MSE  = mean_squared_error(y_test,y_test_pred)

ridge_model = Ridge(alpha =1.0)
ridge_model.fit(x_train_scaled,y_train)

y_test_pred_ridge = ridge_model.predict(x_test_scaled)

ridge_mse  = mean_squared_error(y_test_pred_ridge,y_test)
ridge_r2 = r2_score(y_test,y_test_pred_ridge)

new_house = np.array([[1800,3,3]])
new_house_scaled = scaler.transform(new_house)
predicted_price = lr_model.predict(new_house_scaled)

print("linear regression MSE:",lr_MSE)
print("linear regression r2:",lr_r2)

print("ridge regression MSE:",ridge_mse)
print("ridge regression r2:",ridge_r2)

print("predicted price for new house (lakhs):",predicted_price[0])

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    lr_model,
    x_train_scaled,
    y_train,
    cv=3,
    scoring="r2"
)

print("CV R2:", cv_scores)
print("Mean CV R2:", np.nanmean(cv_scores))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr_model.coef_
})

print("\nFeature Importance:")
print(feature_importance)

residuals = y_test - y_test_pred

print("\nResiduals:")
print(residuals)
print("Mean Residual:", residuals.mean())





