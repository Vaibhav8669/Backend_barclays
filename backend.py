import traceback
import joblib
import pandas as pd
import sys
import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Define allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://barclaysassignment.netlify.app",
    "https://barclays-assignment.onrender.com",
]

print("="*50)
print("🚀 Starting Pre-delinquency API")
print("="*50)
print(f"📋 Allowed Origins: {ALLOWED_ORIGINS}")
print(f"📂 Current Directory: {os.getcwd()}")
print(f"📂 Files in directory: {os.listdir()}")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Add middleware to ensure CORS headers on all responses
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)

    # Get the origin from the request
    origin = request.headers.get("origin")

    # If the origin is allowed, add CORS headers
    if origin in ALLOWED_ORIGINS or (origin and ("netlify.app" in origin or "localhost" in origin)):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type"

    return response

# Handle OPTIONS requests explicitly
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    origin = request.headers.get("origin")

    if origin in ALLOWED_ORIGINS or (origin and ("netlify.app" in origin or "localhost" in origin)):
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600",
            "Access-Control-Expose-Headers": "Content-Length, Content-Type",
        }
        return JSONResponse(content={}, status_code=200, headers=headers)

    return JSONResponse(content={"error": "Origin not allowed"}, status_code=400)

# ---------------- MODEL LOADING ----------------
MODEL_PATH = "pre_delinquency_model.pkl"
print(f"🔍 Looking for model at: {MODEL_PATH}")
print(f"🔍 Model exists: {os.path.exists(MODEL_PATH)}")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model)}")
except Exception as e:
    print("❌ Model failed to load")
    print(f"❌ Error: {str(e)}")
    print(traceback.format_exc())
    model = None

# ---------------- TEST ENDPOINTS ----------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pre-delinquency API",
        "status": "running",
        "model_loaded": model is not None,
        "cors_origins": ALLOWED_ORIGINS,
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.get("/health")
async def health(request: Request):
    """Health check endpoint"""
    origin = request.headers.get("origin")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cors_configured": True,
        "allowed_origins": ALLOWED_ORIGINS,
        "request_origin": origin,
        "python_version": sys.version
    }

@app.get("/test")
async def test_cors(request: Request):
    """Test CORS endpoint"""
    origin = request.headers.get("origin")
    return {
        "message": "CORS is working!",
        "origin": origin,
        "allowed_origins": ALLOWED_ORIGINS,
        "headers": dict(request.headers),
        "method": request.method
    }

@app.get("/debug")
async def debug(request: Request):
    """Debug endpoint"""
    return {
        "client_host": request.client.host if request.client else "unknown",
        "origin": request.headers.get("origin"),
        "headers": dict(request.headers),
        "method": request.method,
        "url": str(request.url),
        "model_loaded": model is not None,
        "python_version": sys.version,
        "cors_origins": ALLOWED_ORIGINS,
        "cwd": os.getcwd(),
        "files": os.listdir()
    }

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
    """
    Compute risk prediction from input features
    """
    df = input_df.copy()

    pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    bill_cols = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
    pay_amt_cols = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]

    # Feature engineering
    df["avg_delay"] = df[pay_cols].mean(axis=1)
    df["delay_trend"] = df["pay_6"] - df["pay_0"]
    df["bill_growth"] = (df["bill_amt6"] - df["bill_amt1"]) / (df["bill_amt1"] + 1)
    df["utilization_avg"] = df[bill_cols].mean(axis=1) / (df["limit_bal"] + 1)
    df["pay_cover_ratio_avg"] = (
            df[pay_amt_cols].mean(axis=1) /
            (df[bill_cols].mean(axis=1) + 1)
    )
    df["cash_flow_proxy"] = df[pay_amt_cols].mean(axis=1)

    # One-hot encoding
    df["sex_2"] = (df["sex"] == 2).astype(int)

    for i in range(1, 7):
        df[f"education_{i}"] = (df["education"] == i).astype(int)

    for i in range(1, 4):
        df[f"marriage_{i}"] = (df["marriage"] == i).astype(int)

    feature_order = [
        "limit_bal", "age", "avg_delay", "delay_trend", "pay_cover_ratio_avg",
        "bill_growth", "utilization_avg", "cash_flow_proxy", "sex_2",
        "education_1", "education_2", "education_3", "education_4",
        "education_5", "education_6", "marriage_1", "marriage_2", "marriage_3",
    ]

    X = df[feature_order]

    if model is None:
        raise Exception("Model not loaded")

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

# ---------------- API ENDPOINTS ----------------
@app.post("/predict")
async def predict(data: CustomerData, request: Request):
    """Single customer prediction endpoint"""
    if model is None:
        return JSONResponse(
            content={"error": "Model not loaded"},
            status_code=503
        )

    try:
        input_df = pd.DataFrame([data.dict()])
        risk_score, level, action, reason = compute_prediction(input_df)

        result = {
            "risk_score": risk_score,
            "risk_level": level,
            "recommended_action": action,
            "reason": reason
        }

        # Create response with CORS headers
        origin = request.headers.get("origin")
        response = JSONResponse(content=result)

        if origin in ALLOWED_ORIGINS or (origin and ("netlify.app" in origin or "localhost" in origin)):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), request: Request):
    """Batch prediction from CSV file"""
    if model is None:
        return JSONResponse(
            content={"error": "Model not loaded"},
            status_code=503
        )

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
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/dashboard-metrics")
async def dashboard_metrics(request: Request):
    """Dashboard metrics endpoint"""
    print("📊 Dashboard metrics requested")

    data = {
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

    # Create response with CORS headers
    origin = request.headers.get("origin")
    response = JSONResponse(content=data)

    if origin in ALLOWED_ORIGINS or (origin and ("netlify.app" in origin or "localhost" in origin)):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        print(f"✅ Added CORS headers for origin: {origin}")
    else:
        print(f"⚠️ Origin not allowed: {origin}")

    return response

@app.get("/customers")
async def get_customers(request: Request):
    """Get customers with risk scores"""
    print("🚨 /customers HIT")
    print(f"📨 Request origin: {request.headers.get('origin')}")

    raw_customers = [
        {
            "id": 1,
            "name": "Rohan Sharma",
            "accountNumber": "ACC-10021",
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
            risk_score, level, _, _ = compute_prediction(df)
            print(f"✅ Prediction successful for customer {c['id']}: {risk_score}")
        except Exception as e:
            print(f"❌ ML FAILED for customer {c['id']}: {e}")
            risk_score = 0.5
            level = "MEDIUM RISK"

        customers.append({
            "id": c["id"],
            "name": c["name"],
            "accountNumber": c["accountNumber"],
            "riskScore": round(risk_score, 2),
            "riskBucket": level.replace(" RISK", ""),
            "utilizationRate": round(c["bill_amt6"] / c["limit_bal"], 2),
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

    # Create response with CORS headers
    origin = request.headers.get("origin")
    response = JSONResponse(content=customers)

    if origin in ALLOWED_ORIGINS or (origin and ("netlify.app" in origin or "localhost" in origin)):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        print(f"✅ Added CORS headers for origin: {origin}")
    else:
        print(f"⚠️ Origin not allowed: {origin}")

    print(f"📊 Returning {len(customers)} customers")
    return response

# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)