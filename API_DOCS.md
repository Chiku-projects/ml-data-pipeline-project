\# API Documentation



\## Base URL



http://localhost:8000



\## Endpoints



\### GET /health

Check API status



```bash

curl http://localhost:8000/health

```



Response:

```json

{

&#x20; "status": "healthy",

&#x20; "models\_loaded": {

&#x20;   "churn": true,

&#x20;   "ltv": true,

&#x20;   "recommendations": true

&#x20; }

}

```



\### POST /predict/churn

Predict churn probability



\### POST /predict/ltv

Predict lifetime value



\### POST /predict/recommendations

Get product recommendations



\### POST /predict/all

Get all predictions



\## Request Format



```json

{

&#x20; "customer\_id": "C00001",

&#x20; "total\_spending": 1000,

&#x20; "total\_orders": 10,

&#x20; "avg\_order\_value": 100,

&#x20; "days\_since\_last\_purchase": 30,

&#x20; "customer\_tenure\_days": 365,

&#x20; "distinct\_categories\_purchased": 5,

&#x20; "completed\_orders": 10,

&#x20; "cancelled\_orders": 1,

&#x20; "completion\_rate": 90

}

```



\## Response Format



```json

{

&#x20; "customer\_id": "C00001",

&#x20; "churn": {

&#x20;   "churn\_probability": 0.0002,

&#x20;   "churn\_risk": "Low"

&#x20; },

&#x20; "ltv": {

&#x20;   "predicted\_ltv": 891.48,

&#x20;   "segment": "Medium"

&#x20; },

&#x20; "recommendations": {

&#x20;   "recommended\_products": \["P001", "P045", "P078"],

&#x20;   "confidence": 0.85

&#x20; }

}

```



\## Interactive Documentation



Visit: http://localhost:8000/docs (Swagger UI)

