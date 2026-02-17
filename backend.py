import traceback
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- MODEL LOADING ----------------

try:
    model = joblib.load("pre_delinquency_model.pkl")
    print("Model loaded successfully!")
except Exception:
    print("Model failed to load")
    print(traceback.format_exc())
    model = None


# ---------------- INPUT SCHEMA ----------------

class CustomerData(BaseModel):
    limit_bal: float
    sex: int
    education: int
    marriage: int
    age: int

    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float


# ---------------- FEATURE ENGINEERING ----------------

def compute_prediction(input_df: pd.DataFrame):

    df = input_df.copy()

    pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    bill_cols = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
    pay_amt_cols = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]

    df["avg_delay"] = df[pay_cols].mean(axis=1)
    df["delay_trend"] = df["pay_6"] - df["pay_0"]

    df["bill_growth"] = (df["bill_amt6"] - df["bill_amt1"]) / (df["bill_amt1"] + 1)

    df["utilization_avg"] = df[bill_cols].mean(axis=1) / (df["limit_bal"] + 1)

    df["pay_cover_ratio_avg"] = (
        df[pay_amt_cols].mean(axis=1) /
        (df[bill_cols].mean(axis=1) + 1)
    )

    df["cash_flow_proxy"] = df[pay_amt_cols].mean(axis=1)

    # One-hot encoding (MATCH TRAINING FEATURES)
    df["sex_2"] = (df["sex"] == 2).astype(int)

    for i in range(1, 7):
        df[f"education_{i}"] = (df["education"] == i).astype(int)

    for i in range(1, 4):
        df[f"marriage_{i}"] = (df["marriage"] == i).astype(int)

    feature_order = [
        "limit_bal",
        "age",
        "avg_delay",
        "delay_trend",
        "pay_cover_ratio_avg",
        "bill_growth",
        "utilization_avg",
        "cash_flow_proxy",
        "sex_2",
        "education_1",
        "education_2",
        "education_3",
        "education_4",
        "education_5",
        "education_6",
        "marriage_1",
        "marriage_2",
        "marriage_3",
    ]

    X = df[feature_order]

    risk_score = float(model.predict_proba(X)[0][1])

    if risk_score < 0.3:
        level = "LOW RISK"
        action = "Approve normally"
        reason = "Customer shows stable repayment behaviour"
    elif risk_score < 0.7:
        level = "MEDIUM RISK"
        action = "Approve with caution"
        reason = "Customer has moderate risk indicators"
    else:
        level = "HIGH RISK"
        action = "Manual review required"
        reason = "Customer shows strong default signals"

    return risk_score, level, action, reason


# ---------------- SINGLE PREDICTION ----------------

@app.post("/predict")
def predict(data: CustomerData):

    if model is None:
        return {"error": "Model not loaded"}

    try:
        input_df = pd.DataFrame([data.dict()])

        risk_score, level, action, reason = compute_prediction(input_df)

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "recommended_action": action,
            "reason": reason
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------- CSV BATCH PREDICTION ----------------

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    if model is None:
        return {"error": "Model not loaded"}

    try:
        df = pd.read_csv(file.file)

        risk_scores = []
        risk_levels = []

        for _, row in df.iterrows():
            row_df = pd.DataFrame([row])

            risk_score, level, _, _ = compute_prediction(row_df)

            risk_scores.append(risk_score)
            risk_levels.append(level)

        df["risk_score"] = risk_scores
        df["risk_level"] = risk_levels

        output_path = "predictions.csv"
        df.to_csv(output_path, index=False)

        return FileResponse(
            output_path,
            media_type="text/csv",
            filename="predictions.csv"
        )

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


@app.get("/dashboard-metrics")
def dashboard_metrics():
    """
    In real life this would come from DB.
    For now, mock-but-realistic server-side aggregation.
    """

    return {
        "portfolioMetrics": {
            "totalAccounts": 12500,
            "atRiskAccounts": 1840,
            "interventionsActive": 642,
            "preventedDefaults30d": 312,
            "estimatedSavings": 480000
        },
        "riskDistribution": [
            { "bucket": "Low", "count": 8600, "percentage": 69 },
            { "bucket": "Medium", "count": 3260, "percentage": 26 },
            { "bucket": "High", "count": 640, "percentage": 5 }
        ],
        "riskTrendData": [
            { "month": "Nov", "avgRisk": 0.42, "highRisk": 820 },
            { "month": "Dec", "avgRisk": 0.40, "highRisk": 780 },
            { "month": "Jan", "avgRisk": 0.38, "highRisk": 720 },
            { "month": "Feb", "avgRisk": 0.36, "highRisk": 690 },
            { "month": "Mar", "avgRisk": 0.35, "highRisk": 660 },
            { "month": "Apr", "avgRisk": 0.34, "highRisk": 640 }
        ],
        "interventionEffectiveness": [
            { "type": "SMS Reminder", "success": 180, "total": 240 },
            { "type": "Call Center", "success": 140, "total": 190 },
            { "type": "Payment Plan", "success": 95, "total": 120 },
            { "type": "Credit Limit Freeze", "success": 60, "total": 90 }
        ]
    }

@app.get("/customers")
def get_customers():
    print("🚨 /customers HIT")

    raw_customers = [
        {
            "id": 1,
            "name": "Rohan Sharma",
            "accountNumber": "ACC-10021",

            # RAW INPUTS (same schema as /predict)
            "limit_bal": 200000,
            "sex": 2,
            "education": 2,
            "marriage": 1,
            "age": 35,

            "pay_0": 0,
            "pay_2": 0,
            "pay_3": 1,
            "pay_4": 0,
            "pay_5": 0,
            "pay_6": 0,

            "bill_amt1": 50000,
            "bill_amt2": 48000,
            "bill_amt3": 47000,
            "bill_amt4": 46000,
            "bill_amt5": 45000,
            "bill_amt6": 44000,

            "pay_amt1": 5000,
            "pay_amt2": 6000,
            "pay_amt3": 7000,
            "pay_amt4": 6000,
            "pay_amt5": 6000,
            "pay_amt6": 8000
        }
    ]

    customers = []

    for c in raw_customers:
        try:
            # ✅ ONLY pass model input fields
            model_input = {
                k: c[k] for k in [
                    "limit_bal", "sex", "education", "marriage", "age",
                    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
                    "bill_amt1", "bill_amt2", "bill_amt3",
                    "bill_amt4", "bill_amt5", "bill_amt6",
                    "pay_amt1", "pay_amt2", "pay_amt3",
                    "pay_amt4", "pay_amt5", "pay_amt6",
                ]
            }

            df = pd.DataFrame([model_input])

            # 🔍 Debug: PROVE feature alignment
            print("📊 Input columns:", df.columns.tolist())

            risk_score, level, _, _ = compute_prediction(df)

        except Exception as e:
            # 🛡 Never crash endpoint
            print("❌ ML FAILED:", e)
            risk_score = 0.5
            level = "MEDIUM RISK"

        customers.append({
            "id": c["id"],
            "name": c["name"],
            "accountNumber": c["accountNumber"],
            "riskScore": round(risk_score, 2),
            "riskBucket": level.replace(" RISK", ""),
            "utilizationRate": c["bill_amt6"] / c["limit_bal"],
            "currentBalance": c["bill_amt6"],
            "creditLimit": c["limit_bal"],
            "averagePaymentDelay": int(
                sum([c["pay_0"], c["pay_2"], c["pay_3"], c["pay_4"], c["pay_5"], c["pay_6"]]) / 6
            ),
            "daysSinceLastPayment": 18,
            "paymentCoverageRatio": 0.65,
            "trend": "down",
            "behaviorFlags": ["Late payment history"]
        })

    return customers
