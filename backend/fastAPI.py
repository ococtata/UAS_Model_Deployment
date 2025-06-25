from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware

try:
    with open("./model/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open('./model/preprocessing_objects.pkl', 'rb') as f:
        preprocessing = pickle.load(f)
    print("Model and preprocessing objects loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ObesityPredict(BaseModel):
    Age: float
    Height: float
    Weight: float
    FCVC: float
    NCP: float
    CH2O: float
    FAF: float
    TUE: float
    
    Gender: str
    family_history_with_overweight: str
    FAVC: str
    SMOKE: str
    SCC: str
    MTRANS: str
    CAEC: str
    CALC: str


def preprocess_input(data: ObesityPredict):
    try:
        input_dict = {
            'Age': [data.Age],
            'Height': [data.Height], 
            'Weight': [data.Weight],
            'FCVC': [data.FCVC],
            'NCP': [data.NCP],
            'CH2O': [data.CH2O],
            'FAF': [data.FAF],
            'TUE': [data.TUE],
            'Gender': [data.Gender.lower().strip()],
            'family_history_with_overweight': [data.family_history_with_overweight.lower().strip()],
            'FAVC': [data.FAVC.lower().strip()],
            'SMOKE': [data.SMOKE.lower().strip()],
            'SCC': [data.SCC.lower().strip()],
            'MTRANS': [data.MTRANS.lower().strip()],
            'CAEC': [data.CAEC.lower().strip()],
            'CALC': [data.CALC.lower().strip()]
        }
        
        df = pd.DataFrame(input_dict)
        
        min_max_scaler = preprocessing['min_max_scaler']
        oh_encoder = preprocessing['oh_encoder']
        caec_encoder = preprocessing.get('caec_encoder')
        calc_encoder = preprocessing.get('calc_encoder')
        numerical_cols = preprocessing['numerical_cols']
        non_hierarchical_cols = preprocessing['non_hierarchical_cols']
        
        df[numerical_cols] = min_max_scaler.transform(df[numerical_cols])
        
        oh_encoded = oh_encoder.transform(df[non_hierarchical_cols])
        oh_columns = oh_encoder.get_feature_names_out(non_hierarchical_cols)
        oh_df = pd.DataFrame(oh_encoded, columns=oh_columns, index=df.index)
        
        if caec_encoder and 'CAEC' in df.columns:
            df['CAEC'] = caec_encoder.transform(df[['CAEC']]).flatten()
            
        if calc_encoder and 'CALC' in df.columns:
            df['CALC'] = calc_encoder.transform(df[['CALC']]).flatten()
        
        df = pd.concat([df.drop(columns=non_hierarchical_cols), oh_df], axis=1)

        expected_order = [
            'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE', 'CALC',
            
            'Gender_female', 'Gender_male', 
            'family_history_with_overweight_no', 'family_history_with_overweight_yes', 
            'FAVC_no', 'FAVC_yes', 
            'SMOKE_no', 'SMOKE_yes', 
            'SCC_no', 'SCC_yes', 
            'MTRANS_automobile', 'MTRANS_bike', 'MTRANS_motorbike', 
            'MTRANS_public_transportation', 'MTRANS_walking'
        ]
        
        missing_cols = [col for col in expected_order if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        df = df[expected_order]
        
        return df
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

@app.get("/")
def status():
    return {
        "message": "Obesity prediction API is live!",
        "model_type": type(model).__name__,
        "status": "ready"
    }

@app.post("/predict")
def predict(data: ObesityPredict):
    try:
        processed_data = preprocess_input(data)
        
        prediction_encoded = model.predict(processed_data)[0]
        
        target_encoder = preprocessing['target_encoder']
        prediction = target_encoder.inverse_transform([[prediction_encoded]])[0][0]
        
        prediction_proba = model.predict_proba(processed_data)[0]
        
        obesity_classes = preprocessing['obesity_order']
        probabilities = {obesity_classes[i]: float(prob) for i, prob in enumerate(prediction_proba)}
        
        confidence = float(max(prediction_proba))
        
        return {
            "predicted_obesity_level": prediction,
            "confidence": round(confidence * 100, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()},
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")