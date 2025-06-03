import joblib
from models.full_code import LinearRegression, ridge_regression

# Load mô hình từ file
def load_model(path):
    with open(path, 'rb') as f:
        model = joblib.load(f)
    return model

# Hàm dự đoán
def predict(model, input_data):
    predicted = model.predict(input_data)
    return predicted

def predict_formula(model):
    poly = model.poly
    selector = model.selector
    feature_names = poly.get_feature_names_out(input_features=["Input", "Output", "Interface", "Enquiry", "NPDR_AFP", "File", "Duration"])
    selector_mask = selector.get_support()
    selected_features = ["*".join(feature.split(" ")) for feature in feature_names[selector_mask]]
    selected_features = [ *selected_features, "Resource" ] 
    y = f"{model.beta[0, 0]}"
    for i in range(1, len(model.beta)):
        y += f" + {round(model.beta[i, 0], 2)}*{selected_features[i-1]}"
    return y