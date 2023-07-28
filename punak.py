Punak_dataset = [data['Address'].map(lambda x :x == 'Punak')]
desire_features = ['Area', 'Room','Parking', 'Warehouse', 'Elevator']
x_dataset = Punak_dataset[desire_features].values
y_dataset = Punak_dataset['Price'].values
x_dataset = x_dataset.astype(np.float64)
y_dataset = y_dataset.astype(np.float64)
x_train,x_test,y_train,y_test = train_test_split(x_dataset, y_dataset, test_size= 0.2)
lls_model = LLS()
lls_model.fit(x_train,y_train)
Y_pred = lls_model.predict(x_test)

print(Y_pred)
# Calculate MAE
mae_p = mean_absolute_error(y_test, Y_pred)
print("(MAE):", mae_p)

# Calculate MSE
mse_p = mean_squared_error(y_test, Y_pred)
print("(MSE):", mse_p)

# Calculate RMSE
rmse_p = np.sqrt(mse_p)
print("(RMSE):", rmse_p)