import joblib
from models.full_code import LinearRegression, ridge_regression

# Load mô hình từ file
def load_model(path):
    with open(path, 'rb') as f:
        model_package = joblib.load(f)
    return model_package

# Hàm dự đoán
def predict(model_package, input_data):
    model = model_package["model"]
    # selector = model_package["selector"]
    # poly_transformer = model_package["poly_transformer"]
    data_poly = model.poly_transformer.transform(input_data)
    if model.selector is not None:
       data_poly = model.selector.transform(data_poly)
    predicted = model.predict(data_poly)
    return predicted