\# Setup Guide



\## Prerequisites



\- Python 3.12+

\- Windows 10/11

\- Git

\- \~2 GB disk space



\## Installation



\### Step 1: Install Dependencies



```bash

pip install -r requirements.txt

```



\### Step 2: Start FastAPI



```bash

cd api

python main.py

```



Visit: http://localhost:8000/docs



\### Step 3: Start Dashboard



```bash

cd dashboard

streamlit run dashboard.py

```



Visit: http://localhost:8501



\## Configuration



\### Databricks



Set environment variables:

DATABRICKS\_HOST=your-host

DATABRICKS\_TOKEN=your-tok

\### API



Default: Port 8000, localhost



\## Testing



\### Health Check



```bash

curl http://localhost:8000/health

```



\### Make Prediction



```bash

curl -X POST http://localhost:8000/predict/all \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"customer\_id":"C001","total\_spending":1000,...}'

```



\## Troubleshooting



\### Port Already in Use



```bash

\# Find process

netstat -ano | findstr :8000



\# Kill it

taskkill /PID <PID> /F

```



\### Models Not Loading



Check: `ml\_models/models/` folder exists



\---



Setup Complete! 🎉

