# from exp import *
# import streamlit as st
# os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

# def use_models():
#     try:
#         with st.spinner("Loading models..."):
#             os.makedirs(MODELS_DIR, exist_ok=True)

#             model_components = {
#                 'xgb': ['xgb_model.pkl', 'symptom_features.pkl'],
#                 'clinical': ['clinical_model.pkl', 'clinical_features.pkl', 'clinical_scaler.pkl'],
#                 'env': ['env_model.pkl', 'env_features.pkl'],
#                 'bert': ['bert_model.pkl'],
#                 'shared': ['disease_encoder.pkl']
#             }

#             # Check which components exist
#             existing_components = {}
#             for name, files in model_components.items():
#                 existing_components[name] = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)

#             # Special BERT check (needs tokenizer and model directories)
#             bert_ready = (os.path.exists(os.path.join(MODELS_DIR, "bert_tokenizer")) and
#                         os.path.exists(os.path.join(MODELS_DIR, "bert_model")))
#             existing_components['bert'] = existing_components['bert'] and bert_ready

#             # Check if all components exist
#             all_ready = all(existing_components.values())
            
#             if not all_ready:
#                 st.warning("Some models are missing. Training needed before predictions can be made.")
                
#                 if st.button("Train Models Now"):
#                     with st.spinner("Training models (this may take several minutes)..."):
#                         success = train_and_save_models()
#                         if not success:
#                             st.error("Model training failed. Please check logs.")
#                             return None
#                 else:
#                     return None

#             # Load all models
#             models = (
#                 joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'env_model.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'bert_model.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'disease_encoder.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'symptom_features.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'clinical_features.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'env_features.pkl')),
#                 joblib.load(os.path.join(MODELS_DIR, 'clinical_scaler.pkl'))
#             )
#             return models

#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         return None
    

# def live():
#     st.set_page_config(
#         page_title="Multi-Modal Disease Prediction System",
#         page_icon="🧬",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

    
#     # Set up main page
#     st.title("🧬 Multi-Modal Disease Prediction System")
#     st.markdown("""
#     This system uses a combination of symptom analysis, clinical data, and environmental factors to predict potential diseases.
#     It employs multiple machine learning models including XGBoost, Logistic Regression, Random Forest, Clinical BERT, and ensemble techniques.
#     """)
    
#     with st.sidebar:
#         st.header("Navigation")
#         app_mode = st.radio(
#             "Select a mode:",
#             ["Prediction","History",]
            
#         )
#     if st.button("Logout 🔒"):
#             st.session_state.clear()
#             register_page()
#             st.success("Logged out successfully!")

#     if st.session_state.get("logged_in"):
#          username = st.session_state["username"]

#     if app_mode == "Prediction":
#         # 🔮 Your existing prediction UI and logic here
#         st.header("🧠 AI Disease Prediction System")
#         # include your form, prediction logic, etc.
#     elif app_mode == "History":
#        st.header("📜 Your Prediction History")

#     # # Fetch predictions for the current user
#     user_predictions = prediction_collection.find({"username": username}).sort("timestamp", -1)

#     table_data = []

#     for record in user_predictions:
#         timestamp = record.get("timestamp")
#         symptoms = record.get("symptoms", "")

#         # Safely access 'predictions' and its fields
#         predictions = record.get("predictions", {})
#         Disease = predictions.get("primary_prediction", "")
#         primary_confidence = predictions.get("primary_confidence", 0)
#         primary_risk = predictions.get("primary_risk", 0)

#         row = {
#             "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "",
#             "Symptoms": symptoms,
#             "Prediction": Disease,
#             "Confidence": round(primary_confidence * 100, 2),
#             "Risk (%)": round(primary_risk * 100, 2)
#         }

#         table_data.append(row)

#     if table_data:
#         st.dataframe(table_data, use_container_width=True)

#         # Optional: show full prediction details in expanders
#         for idx, record in enumerate(user_predictions):
#             with st.expander(f"🔍 Details for Prediction #{idx+1}"):
#                 predictions = record.get("predictions", {})
#                 all_preds = predictions.get("all_predictions", [])

#                 for i, pred in enumerate(all_preds):
#                     st.markdown(f"**Disease #{i+1}:** {pred.get('disease', 'N/A')}")
#                     st.write({
#                         "Confidence": round(pred.get("confidence", 0) * 100, 2),
#                         "Risk %": round(pred.get("risk_percentage", 0) * 100, 2),
#                         "Symptoms Matched": pred.get("symptom_matches", 0),
#                         "Severity Factor": pred.get("severity_factor", 0),
#                         "Warning": pred.get("warning", "None")
#                     })
#                     st.markdown("---")

#     else:
#         st.info("No prediction history found for this user.")


    
    
#     # Load models
#     models = use_models()
    
#     if app_mode == "Prediction":
#         if models is None:
#             st.warning("Please train the models first before making predictions.")
#             return
            
#         st.header("Disease Prediction")
        
#         st.subheader("Patient Information")

#         # Symptoms Section
#         st.markdown("### Symptom Information")
#         symptoms = st.text_area(
#             "Describe your symptoms:", 
#             placeholder="e.g., fever, headache, cough, runny nose",
#             help="Enter all symptoms you are experiencing, separated by commas."
#         )

#         # Clinical Data Section
#         st.markdown("### Clinical Data")
#         col1, col2 = st.columns(2)

#         with col1:
#             age = st.slider("Age", 1, 100, 35, help="Age in years")
#             weight = st.slider("Weight (kg)", 30, 150, 70, help="Weight in kilograms")
#             bp = st.slider("Blood Pressure (systolic)", 60, 200, 120, help="Systolic blood pressure in mmHg")
#             sugar = st.slider("Blood Sugar", 50, 300, 100, help="Fasting blood sugar in mg/dL")

#         with col2:
#             cholesterol = st.slider("Cholesterol", 10, 300, 150, help="Total cholesterol in mg/dL")
#             wbc = st.slider("White Blood Cell Count", 2.0, 20.0, 7.0, step=0.1, help="WBC count in thousands/μL")
#             bmi = st.slider("BMI", 15.0, 40.0, 24.5, step=0.1, help="Body Mass Index")
#             sleep = st.slider("Sleep (hours/day)", 0, 12, 7, help="Average sleep duration in hours")

#         # Environmental Factors Section
#         st.markdown("### Environmental Factors")
#         col1, col2 = st.columns(2)

#         with col1:
#             temperature = st.selectbox("Temperature", ["low", "medium", "high"], index=1)
#             humidity = st.selectbox("Humidity", ["low", "medium", "high"], index=1)
#             air_quality = st.selectbox("Air Quality", ["bad", "normal", "good"], index=1)
#             water_quality = st.selectbox("Water Quality", ["bad", "normal", "good"], index=1)

#         with col2:
#             region_types = list(ENV_MAPPINGS['region_type'].keys())
#             region_types = [r for r in region_types if isinstance(r, str) and r != "None"]
#             region_type = st.selectbox("Region Type", region_types, index=region_types.index("urban"))
            
#             weather_types = list(ENV_MAPPINGS['weather'].keys())
#             weather_types = [w for w in weather_types if isinstance(w, str) and w != "None"]
#             weather = st.selectbox("Weather", weather_types, index=weather_types.index("sunny"))
            
#             time_delay = st.selectbox(
#                 "When did symptoms first appear?",
#                 ["recent (<5 days)", "moderate (5-15 days)", "long (>15 days)"],
#                 index=0
#             )
            
#             # Map selections to time delay codes
#             time_mapping = {
#                 "recent (<5 days)": 28,
#                 "moderate (5-15 days)": 29,
#                 "long (>15 days)": 30
#             }

#         # Create predict button
#         if st.button("Predict Disease", type="primary"):
#             if not symptoms:
#                 st.error("Please enter your symptoms to make a prediction.")
#                 return
                
#             with st.spinner("Analyzing your health data..."):
#                 try:
#                     # Prepare clinical data
#                     clinical_data = {
#                         'Age': age,
#                         'Weight': weight,
#                         'BP': bp,
#                         'Sugar': sugar,
#                         'Cholesterol': cholesterol,
#                         'WBC': wbc,
#                         'BMI': bmi,
#                         'Sleep': sleep
#                     }
                    
#                     # Prepare environmental data
#                     env_data = {
#                         'temperature': ENV_MAPPINGS['temperature'][temperature],
#                         'humidity': ENV_MAPPINGS['humidity'][humidity],
#                         'air_quality': ENV_MAPPINGS['air_quality'][air_quality],
#                         'water_quality': ENV_MAPPINGS['water_quality'][water_quality],
#                         'region_type': ENV_MAPPINGS['region_type'][region_type],
#                         'weather': ENV_MAPPINGS['weather'][weather],
#                         'time_delay': time_mapping[time_delay]
#                     }
                    
#                     # Make prediction
#                     predictions = predict_disease(symptoms, clinical_data, env_data, models)
                    
#                     # Display results
#                     st.header("Prediction Results")
                    
#                     for i, pred in enumerate(predictions):
#                         col1, col2 = st.columns([1, 3])
                        
#                         with col1:
#                             if i == 0:
#                                 st.metric("Primary Prediction", pred['disease'], f"{pred['confidence']:.1%}")
#                             else:
#                                 st.metric(f"Alternative {i}", pred['disease'], f"{pred['confidence']:.1%}")
                        
#                         with col2:
#                             # Create risk visualization
#                             risk_color = "red" if pred['risk_percentage'] > 0.6 else "orange" if pred['risk_percentage'] > 0.4 else "green"
#                             risk_text = "High" if pred['risk_percentage'] > 0.6 else "Moderate" if pred['risk_percentage'] > 0.4 else "Low"
                            
#                             st.markdown(f"""
#                             **Risk Assessment:** {risk_text} ({pred['risk_percentage']:.1%})
#                             <div style="width:100%; background-color:#f0f0f0; height:20px; border-radius:5px;">
#                               <div style="width:{pred['risk_percentage']*100}%; background-color:{risk_color}; height:20px; border-radius:5px;"></div>
#                             </div>
#                             """, unsafe_allow_html=True)
                            
#                             if pred['warning']:
#                                 st.warning(pred['warning'])
                            
#                     st.info("Please note: This system provides predictions based on the information provided and should not replace professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.")
                    
#                 except Exception as e:
#                     st.error(f"Error making prediction: {str(e)}")
    
#     elif app_mode == "Model Information":
#         st.header("Model Information")
        
#         st.markdown("""
#         ### Multiple Model Approach
        
#         This system uses an ensemble of four specialized models:
        
#         1. **Symptom Model** (XGBoost) - Analyzes reported symptoms
#         2. **Clinical Model** (Logistic Regression) - Processes clinical measurements
#         3. **Environmental Model** (Random Forest) - Evaluates environmental factors
#         4. **Language Model** (Clinical BERT) - Uses natural language processing to understand symptom descriptions
        
#         ### Features Used
#         """)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Clinical Features")
#             st.markdown("""
#             - Age
#             - Weight
#             - Blood Pressure
#             - Blood Sugar
#             - Cholesterol
#             - White Blood Cell Count
#             - BMI
#             - Sleep Duration
#             """)
            
#         with col2:
#             st.subheader("Environmental Features")
#             st.markdown("""
#             - Temperature
#             - Humidity
#             - Air Quality
#             - Water Quality
#             - Region Type
#             - Weather Conditions
#             - Time Since Symptoms Began
#             """)
        
#         st.subheader("Risk Assessment")
#         st.markdown("""
#         The system calculates a risk multiplier based on:
#         - Number of matching symptoms
#         - Clinical indicators (abnormal values)
#         - Environmental risk factors
#         - Time delay in seeking care
        
#         This produces a personalized risk score that adapts to your specific situation.
#         """)

#         # 🟩 Final Prediction is ready — now save to MongoDB
#     if st.session_state.get("logged_in"):
#         username = st.session_state["username"]
#         user = user_collection.find_one({"username": username})
#         pred=prediction_collection.find()
#         if user:
#             prediction_data = {
#                 "username": username,
#                 "mobile": user.get("mobile"),
#                 "timestamp": datetime.datetime.utcnow(),
#                 "symptoms": symptoms,   
#                "predictions": {
#                  "all_predictions": prediction_collection,
                     
#                 }

#             }

#             prediction_collection.insert_one(prediction_data)
#             st.success("✅ Prediction result saved successfully.")
#         else:
#             st.warning("⚠️ User data not found for saving prediction.")



# import streamlit as st
# from pymongo import MongoClient
# import datetime


# # --- MongoDB Connection ---
# MONGO_URI = "mongodb://localhost:27017/"
# client = MongoClient(MONGO_URI)
# db = client["local_db"]
# user_collection = db["users"]
# form_collection = db["user_info"]
# prediction_collection = db["predictions"]


# # --- Session State ---
# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False

# if "username" not in st.session_state:
#     st.session_state["username"] = ""

# if "page" not in st.session_state:
#     st.session_state["page"] = "login"

# # --- Helper Functions ---
# def switch_page(page_name):
#     st.session_state["page"] = page_name
#     st.rerun()


# # --- Registration Page ---
# def register_page():
#     st.title("🆕 Register")

#     username = st.text_input("Choose a Username")
#     password = st.text_input("Choose a Password", type="password")
#     mobileNumber=st.text_input("Enter your number")
#     email=st.text_input("enter your email")
#     register_btn = st.button("Register")

#     if register_btn:
#         if user_collection.find_one({"username": username}):
#             st.error("⚠ Username already exists. Please login.")
#         elif not username or not password:
#             st.warning("⚠ Please fill in all fields.")
#         else:
#             user_collection.insert_one({"username": username, "password": password,"mobileNumber":mobileNumber,"email":email})
#             st.success("✅ Registered successfully! Please login.")
#             switch_page("login")

#     st.button("Already have an account? Login", on_click=lambda: switch_page("login"))

# # --- Login Page ---
# def login_page():
#     st.title("🔐 Login")

#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     login_btn = st.button("Login")

#     if login_btn:
#         user = user_collection.find_one({"username": username, "password": password})
#         if user:
#             st.session_state["logged_in"] = True
#             st.session_state["username"] = username
#             switch_page("main")  # optional if you want multiple app sections


#         else:
#             st.error("❌ Invalid username or password.")

#     st.button("New user? Register here", on_click=lambda: switch_page("register"))

# # --- Main App Page ---
# def main_page():
#     st.title(f"📋 Welcome, {st.session_state['username']}")
    
#         # Example input fields
#     symptoms = st.text_area("Enter Symptoms")
#     clinical_data = st.text_input("Enter Clinical Data (e.g., BP, Sugar, etc.)")
#     environment = st.text_input("Enter Environmental Description")

#     if st.button("Predict"):
#         # Assume you call your model and get this result:
#         prediction_result = "High Risk of Asthma"  # Replace with actual prediction

#         st.success(f"🩺 Prediction: {prediction_result}")

#         # Get current user info from DB
#         user_info = user_collection.find_one({"username": st.session_state["username"]})

#         # Save the input and result
#         prediction_collection.insert_one({
#             "username": st.session_state["username"],
#             "phone": user_info["phone"],
#             "email": user_info["email"],
#             "symptoms": symptoms,
#             "clinical_data": clinical_data,
#             "environment": environment,
#             "prediction": prediction_result,
#             "timestamp": datetime.datetime.now()
#         })


# # --- Page Routing ---
# if st.session_state["logged_in"]:
#      live()  # Redirect to main disease prediction app
# else:
#     if st.session_state["page"] == "login":
#         login_page()
#     elif st.session_state["page"] == "register":
#         register_page()




            # from exp import *
            # import streamlit as st
            # import os
            # import datetime
            # from pymongo import MongoClient
            # import joblib

            # # --- MongoDB Connection ---
            # MONGO_URI = "mongodb://localhost:27017/"
            # client = MongoClient(MONGO_URI)
            # db = client["local_db"]
            # user_collection = db["users"]
            # prediction_collection = db["predictions"]

            # # --- Session State ---
            # if "logged_in" not in st.session_state:
            #     st.session_state["logged_in"] = False

            # if "username" not in st.session_state:
            #     st.session_state["username"] = ""

            # if "page" not in st.session_state:
            #     st.session_state["page"] = "login"

            # # --- Page Switcher ---
            # def switch_page(page_name):
            #     st.session_state["page"] = page_name
            #     st.rerun()

            # # --- Registration Page ---
            # def register_page():
            #     st.title("🆕 Register")
            #     username = st.text_input("Choose a Username")
            #     password = st.text_input("Choose a Password", type="password")
            #     mobile = st.text_input("Enter your mobile number")
            #     email = st.text_input("Enter your email")
            #     if st.button("Register"):
            #         if user_collection.find_one({"username": username}):
            #             st.error("Username already exists.")
            #         elif not username or not password:
            #             st.warning("Fill in all fields.")
            #         else:
            #             user_collection.insert_one({"username": username, "password": password, "mobile": mobile, "email": email})
            #             st.success("Registered successfully! Please login.")
            #             switch_page("login")
            #     st.button("Already have an account? Login", on_click=lambda: switch_page("login"))

            # # --- Login Page ---
            # def login_page():
            #     st.title("🔐 Login")
            #     username = st.text_input("Username")
            #     password = st.text_input("Password", type="password")
            #     if st.button("Login"):
            #         user = user_collection.find_one({"username": username, "password": password})
            #         if user:
            #             st.session_state["logged_in"] = True
            #             st.session_state["username"] = username
            #             switch_page("main")
            #         else:
            #             st.error("Invalid credentials.")
            #     st.button("New user? Register here", on_click=lambda: switch_page("register"))

            # # --- Load Models ---
            # def use_models():
            #     try:
            #         with st.spinner("Loading models..."):
            #             os.makedirs(MODELS_DIR, exist_ok=True)
            #             model_files = [
            #                 'xgb_model.pkl', 'clinical_model.pkl', 'env_model.pkl',
            #                 'disease_encoder.pkl', 'symptom_features.pkl', 'clinical_features.pkl',
            #                 'env_features.pkl', 'clinical_scaler.pkl', 'symptom_data.pkl'
            #             ]
            #             if not all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in model_files):
            #                 if st.button("Train Models Now"):
            #                     if not train_and_save_models():
            #                         st.error("Model training failed.")
            #                         return None
            #             return tuple(joblib.load(os.path.join(MODELS_DIR, f)) for f in model_files)
            #     except Exception as e:
            #         st.error(f"Error loading models: {e}")
            #         return None

            # # --- Main App ---
            # def live():
            #     st.set_page_config(page_title="AI Disease Prediction", layout="wide")
            #     st.title("🧬 Disease Risk Prediction System")
            #     st.markdown("Combines symptom, clinical, and environmental data for accurate predictions.")

            #     with st.sidebar:
            #         app_mode = st.radio("Navigation", ["Prediction", "History"])
            #         if st.button("Logout"):
            #             st.session_state.clear()
            #             switch_page("login")

            #     models = use_models()
            #     if not models:
            #         return

            #     if app_mode == "Prediction":
            #         st.header("📋 Enter Your Health Info")
            #         symptoms = st.text_area("Symptoms (comma-separated):")
            #         col1, col2 = st.columns(2)
            #         with col1:
            #             age = st.slider("Age", 1, 100, 30)
            #             weight = st.slider("Weight (kg)", 30, 150, 60)
            #             bp = st.slider("BP", 60, 200, 120)
            #             sugar = st.slider("Sugar", 50, 300, 100)
            #         with col2:
            #             cholesterol = st.slider("Cholesterol", 10, 300, 180)
            #             wbc = st.slider("WBC", 2, 20, 6)
            #             bmi = st.slider("BMI", 5.0, 40.0, 22.0)
            #             sleep = st.slider("Sleep (hrs)", 0, 12, 7)

            #         st.subheader("🌍 Environmental Factors")
            #         col1, col2 = st.columns(2)
            #         with col1:
            #             temp = st.selectbox("Temperature", ['low', 'medium', 'high'])
            #             humidity = st.selectbox("Humidity", ['low', 'medium', 'high'])
            #             air = st.selectbox("Air Quality", ['bad', 'normal', 'good'])
            #             water = st.selectbox("Water Quality", ['bad', 'normal', 'good'])
            #         with col2:
            #             region = st.selectbox("Region Type", list(ENV_MAPPINGS['region_type'].keys()))
            #             weather = st.selectbox("Weather", list(ENV_MAPPINGS['weather'].keys()))
            #             delay = st.selectbox("Symptom Duration", ['recent', 'moderate', 'long'])

            #         if st.button("Predict"):
            #             clinical = {'Age': age, 'Weight': weight, 'BP': bp, 'Sugar': sugar,
            #                         'Cholesterol': cholesterol, 'WBC': wbc, 'BMI': bmi, 'Sleep': sleep}
            #             environmental = {
            #                 'temperature': ENV_MAPPINGS['temperature'][temp],
            #                 'humidity': ENV_MAPPINGS['humidity'][humidity],
            #                 'air_quality': ENV_MAPPINGS['air_quality'][air],
            #                 'water_quality': ENV_MAPPINGS['water_quality'][water],
            #                 'region_type': ENV_MAPPINGS['region_type'][region],
            #                 'weather': ENV_MAPPINGS['weather'][weather],
            #                 'time_delay': ENV_MAPPINGS['time_delay'][delay][0]
            #             }
            #             with st.spinner("Analyzing..."):
            #                 try:
            #                     predictions, matched, unmatched = predict_disease(symptoms, clinical, environmental, models)
            #                     st.header("🩺 Prediction Results")
            #                     for pred in predictions:
            #                         st.subheader(f"{pred['disease']}")
            #                         st.write(f"Risk: {pred['risk_percentage']:.1f}%")
            #                         if pred['warning']:
            #                             st.warning(pred['warning'])
            #                     prediction_collection.insert_one({
            #                         "username": st.session_state["username"],
            #                         "timestamp": datetime.datetime.utcnow(),
            #                         "symptoms": symptoms,
            #                         "predictions": {"all_predictions": predictions}
            #                     })
            #                 except Exception as e:
            #                     st.error(f"Prediction error: {e}")

            #     elif app_mode == "History":
            #         st.header("📜 Your Prediction History")
            #         records = prediction_collection.find({"username": st.session_state["username"]}).sort("timestamp", -1)
            #         for record in records:
            #             with st.expander(record['timestamp'].strftime("%Y-%m-%d %H:%M:%S")):
            #                 st.write(f"**Symptoms:** {record.get('symptoms', '')}")
            #                 for pred in record['predictions'].get('all_predictions', []):
            #                     st.write(f"- {pred['disease']} ({pred['risk_percentage']:.1f}%)")

            # # --- App Entry ---
            # if st.session_state["logged_in"]:
            #     live()
            # elif st.session_state["page"] == "login":
            #     login_page()
            # elif st.session_state["page"] == "register":
            #     register_page()







from exp import *
import streamlit as st
import os
import datetime
from pymongo import MongoClient
import joblib
import pandas as pd

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["local_db"]
user_collection = db["users"]
prediction_collection = db["predictions"]

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""

if "page" not in st.session_state:
    st.session_state["page"] = "login"

# --- Page Switcher ---
def switch_page(page_name):
    st.session_state["page"] = page_name
    st.rerun()

# --- Registration Page ---
def register_page():
    st.title("🆕 Register")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    mobile = st.text_input("Enter your mobile number")
    email = st.text_input("Enter your email")
    if st.button("Register"):
        if user_collection.find_one({"username": username}):
            st.error("Username already exists.")
        elif not username or not password:
            st.warning("Fill in all fields.")
        else:
            user_collection.insert_one({"username": username, "password": password, "mobile": mobile, "email": email})
            st.success("Registered successfully! Please login.")
            switch_page("login")
    st.button("Already have an account? Login", on_click=lambda: switch_page("login"))

# --- Login Page ---
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = user_collection.find_one({"username": username, "password": password})
        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            switch_page("main")
        else:
            st.error("Invalid credentials.")
    st.button("New user? Register here", on_click=lambda: switch_page("register"))

# --- Load Models ---
def use_models():
    try:
        with st.spinner("Loading models..."):
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_files = [
                'xgb_model.pkl', 'clinical_model.pkl', 'env_model.pkl',
                'disease_encoder.pkl', 'symptom_features.pkl', 'clinical_features.pkl',
                'env_features.pkl', 'clinical_scaler.pkl', 'symptom_data.pkl'
            ]
            if not all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in model_files):
                if st.button("Train Models Now"):
                    if not train_and_save_models():
                        st.error("Model training failed.")
                        return None
            return tuple(joblib.load(os.path.join(MODELS_DIR, f)) for f in model_files)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# --- History Page ---
def history_page():
    st.header("📜 Your Prediction History")
    records = prediction_collection.find({"username": st.session_state["username"]}).sort("timestamp", -1)
    records_list = list(records)
    
    if len(records_list) == 0:
        st.write("No prediction history found.")
    else:
        # Prepare data for the table
        history_data = []
        for record in records_list:
            # Convert matched symptoms to a string, in case it's a list
            matched_symptoms = ', '.join([str(symptom) for symptom in record['predictions'].get('matched_symptoms', [])])

            # Extract diseases and risk percentages as strings
            predictions = [f"{pred['disease']} (Risk: {pred['risk_percentage']:.1f}%)" for pred in record['predictions'].get("all_predictions", [])]
            
            history_data.append({
                "Timestamp": record['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                "Symptoms": record.get('symptoms', ''),
                "Predictions": ', '.join(predictions),  # Joining predictions into a string
                "Risk (%)": ', '.join([f"{pred['risk_percentage']:.1f}%" for pred in record['predictions'].get("all_predictions", [])]),
                "Matched Symptoms": matched_symptoms
            })
        
        # Convert to DataFrame for easy display in table format
        df = pd.DataFrame(history_data)
        
        # Display the data in a dropdown as a table
        with st.expander("🔍 View History in Table Format"):
            st.dataframe(df)



# --- Main App ---
def live():
    st.set_page_config(page_title="AI Disease Prediction", layout="wide")
    st.title("🧬 Disease Risk Prediction System")
    st.markdown("Combines symptom, clinical, and environmental data for accurate predictions.")

    with st.sidebar:
        app_mode = st.radio("Navigation", ["Prediction", "History"])
        if st.button("Logout"):
            st.session_state.clear()
            switch_page("login")

    models = use_models()
    if not models:
        return

    if app_mode == "Prediction":
        st.header("📋 Enter Your Health Info")
        symptoms = st.text_area("Symptoms (comma-separated):")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 1, 100, 30)
            weight = st.slider("Weight (kg)", 30, 150, 60)
            bp = st.slider("BP", 60, 200, 120)
            sugar = st.slider("Sugar", 50, 300, 100)
        with col2:
            cholesterol = st.slider("Cholesterol", 10, 300, 180)
            wbc = st.slider("WBC", 2, 20, 6)
            bmi = st.slider("BMI", 5.0, 40.0, 22.0)
            sleep = st.slider("Sleep (hrs)", 0, 12, 7)

        st.subheader("🌍 Environmental Factors")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.selectbox("Temperature", ['low', 'medium', 'high'])
            humidity = st.selectbox("Humidity", ['low', 'medium', 'high'])
            air = st.selectbox("Air Quality", ['bad', 'normal', 'good'])
            water = st.selectbox("Water Quality", ['bad', 'normal', 'good'])
        with col2:
            region = st.selectbox("Region Type", list(ENV_MAPPINGS['region_type'].keys()))
            weather = st.selectbox("Weather", list(ENV_MAPPINGS['weather'].keys()))
            delay = st.selectbox("Symptom Duration", ['recent', 'moderate', 'long'])

        if st.button("Predict"):
            clinical = {'Age': age, 'Weight': weight, 'BP': bp, 'Sugar': sugar,
                        'Cholesterol': cholesterol, 'WBC': wbc, 'BMI': bmi, 'Sleep': sleep}
            environmental = {
                'temperature': ENV_MAPPINGS['temperature'][temp],
                'humidity': ENV_MAPPINGS['humidity'][humidity],
                'air_quality': ENV_MAPPINGS['air_quality'][air],
                'water_quality': ENV_MAPPINGS['water_quality'][water],
                'region_type': ENV_MAPPINGS['region_type'][region],
                'weather': ENV_MAPPINGS['weather'][weather],
                'time_delay': ENV_MAPPINGS['time_delay'][delay][0]
            }
            with st.spinner("Analyzing..."):
                try:
                    predictions, matched, unmatched = predict_disease(symptoms, clinical, environmental, models)
                    st.header("🩺 Prediction Results")
                    for pred in predictions:
                        st.subheader(f"{pred['disease']}")
                        st.write(f"Risk: {pred['risk_percentage']:.1f}%")
                        st.write(f"Matched Symptoms: {', '.join(pred['matched_symptom_names'])}")
                        st.write(f"Severity Factor: {pred['severity_factor']:.2f}")
                        if pred['warning']:
                            st.warning(pred['warning'])

                    if matched:
                        with st.expander("🔍 Matched Symptoms"):
                            for sym in matched:
                                st.write(f"Input: {sym[0]} → Matched: {sym[1]} (Score: {sym[2]:.1f})")
                    if unmatched:
                        with st.expander("❌ Unmatched Symptoms"):
                            st.write(", ".join(unmatched))

                    # Save prediction data to MongoDB
                    prediction_collection.insert_one({
                        "username": st.session_state["username"],
                        "timestamp": datetime.datetime.utcnow(),
                        "symptoms": symptoms,
                        "clinical_data": clinical,
                        "environmental_data": environmental,
                        "predictions": {
                            "all_predictions": predictions,
                            "matched_symptoms": matched,
                            "unmatched_symptoms": unmatched
                        }
                    })
                    st.success("✅ Prediction result saved successfully.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    elif app_mode == "History":
        history_page()

# --- App Entry ---
if st.session_state["logged_in"]:
    live()
elif st.session_state["page"] == "login":
    login_page()
elif st.session_state["page"] == "register":
    register_page()
