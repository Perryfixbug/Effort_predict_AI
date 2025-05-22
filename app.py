import streamlit as st
from datetime import datetime

from models.model_utils import load_model, predict

# Load model
model_package = load_model("models/ridge_model.pkl")
model = model_package["model"]

# Streamlit UI config
st.set_page_config(page_title="Effort Predict AI", layout="centered")
st.title("Effort Predict AI")

# Input UI
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", value=datetime.today())
    number_of_people = st.number_input("Number of People", min_value=1, step=1)
    workload = st.number_input("Workload", min_value=0, step=1)
    project_scope = st.number_input("Project Scope", min_value=0, step=1)
    risk = st.number_input("Risk (number 1 - 7)", min_value=0, max_value=7, step=1) #Rủi ro cao nhất là 7

with col2:
    end_date = st.date_input("End Date", value=datetime.today())
    topic = st.number_input("Topic (as number 1 - 7)", min_value=0, max_value=7, step=1)
    team_exp = st.number_input("Team Experience", min_value=0, step=1)
    manage_exp = st.number_input("Management Experience", min_value=0, step=1)

# Nút Dự đoán nằm bên phải
btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1])
with btn_col3:
    predict_btn = st.button("Dự đoán", use_container_width=True)

# Nhân nút
if predict_btn:
    duration = (end_date - start_date).days
    if duration <= 0:
        st.error("End Date phải sau Start Date.")
    else:
        input_data = [[
            number_of_people,
            workload,
            project_scope,
            topic,
            risk,
            team_exp,
            manage_exp,
            duration
        ]]

        effort = predict(model_package, input_data)[0][0]
        st.success("Kết quả dự đoán Effort (giờ):")
        st.metric(label="Effort", value=f"{round(effort, 2)}h")
