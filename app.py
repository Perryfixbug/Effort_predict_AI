import streamlit as st
from datetime import datetime
import pandas as pd

from models.model_utils import load_model, predict, predict_formula

# Load model
model = load_model("models/model.pkl")

# Streamlit UI config
st.set_page_config(page_title="Effort Predict AI", layout="centered")
st.title("Effort Predict AI")

# Input UI
# Khởi tạo session state
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.today().date()
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.today().date()
if "duration" not in st.session_state:
    st.session_state.duration = 0
if "manual_duration" not in st.session_state:
    st.session_state.manual_duration = False  # mặc định là dùng ngày

# Toggle: chọn chế độ nhập Duration
st.checkbox("Nhập thủ công thời gian (Duration)", key="manual_duration")

col1, col2 = st.columns(2)

with col1:
    input = st.number_input("Input", min_value=0, step=1)
    interface = st.number_input("Interface", min_value=0, step=1)
    file = st.number_input("File", min_value=0, step=1)
    npdr_afp = st.number_input("NPDR AFP", min_value=0.0, step=0.1)

    if not st.session_state.manual_duration:
        st.session_state.start_date = st.date_input("Start Date", value=st.session_state.start_date)

with col2:
    output = st.number_input("Output", min_value=0, step=1)
    enquiry = st.number_input("Enquiry", min_value=0, step=1)
    resource = st.number_input("Resource", min_value=0, step=1)

    if not st.session_state.manual_duration:
        st.session_state.end_date = st.date_input("End Date", value=st.session_state.end_date)
        st.session_state.duration = (st.session_state.end_date - st.session_state.start_date).days
    else:
        st.session_state.duration = st.number_input("Duration (tháng)", min_value=1, step=1)

# Nút Dự đoán nằm bên phải
col1, col2 = st.columns([3, 1])
with col1:
    y = predict_formula(model)
    terms = y.split(" + ")

    # Gom 3 biểu thức mỗi dòng
    grouped_lines = []
    for i in range(0, len(terms), 2):
        group = terms[i:i+2]
        line = " + ".join(group)
        if i != 0:
            grouped_lines.append(f"+ {line}")
        else:
            grouped_lines.append(line)

    # Ghép thành code block
    formatted_y = "Effort = " + "\n".join(grouped_lines)
    st.code(formatted_y, language="python")

with col2:
    predict_btn = st.button("Dự đoán", use_container_width=True)

# Xử lý khi nhấn nút
if predict_btn:
    duration = st.session_state.duration

    # Kiểm tra hợp lệ nếu dùng ngày
    if not st.session_state.manual_duration and duration <= 0:
        st.error("End Date phải sau Start Date.")
    else:
        input_data = [{
            "Input": input,
            "Output": output,
            "Interface": interface,
            "Enquiry": enquiry,
            "NPDR_AFP": npdr_afp,
            "File": file,
            "Resource": resource,
            "Duration": duration
        }]
        input_data_df = pd.DataFrame(input_data)

        # Gọi mô hình dự đoán
        effort = predict(model, input_data_df)[0][0]
        st.success("Kết quả dự đoán Effort (giờ):")
        st.metric(label="Effort", value=f"{round(effort, 2)}h")