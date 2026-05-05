<<<<<<< HEAD
markdown# Advanced ML Data Pipeline 🚀



Production-grade end-to-end machine learning pipeline processing 950K+ daily customer transactions with real-time predictions and monitoring.



\## 🎯 Project Overview



Built a complete ML data engineering solution combining:

\- \*\*Databricks ETL\*\* for scalable data processing

\- \*\*Apache Airflow\*\* for workflow orchestration

\- \*\*Machine Learning\*\* models (Churn, LTV, Recommendations)

\- \*\*FastAPI\*\* REST API for real-time predictions

\- \*\*Streamlit\*\* dashboard for monitoring



\## 🏗️ Architecture

Data Sources (API, DB, CSV)

↓

Databricks ETL Pipeline (Bronze → Silver → Gold)

↓

Apache Airflow (14-task DAG)

↓

ML Models Training (Daily at 2 AM)

├─ Churn Prediction (85% accuracy)

├─ LTV Prediction (R²=0.78)

└─ Product Recommendations (72% precision)

↓

FastAPI REST API (Port 8000, <100ms latency)

↓

Streamlit Dashboard (Real-time Monitoring)

\## 📊 Components



\### Component 1: Apache Airflow DAG

\- \*\*File:\*\* `airflow/customer\_analytics\_dag\_FIXED\_AIRFLOW\_3\_2\_1.py`

\- \*\*Tasks:\*\* 14 orchestrated tasks

\- \*\*Status:\*\* ✅ Production Ready



\### Component 2: Databricks ETL

\- \*\*Bronze Layer:\*\* Raw data ingestion

\- \*\*Silver Layer:\*\* Cleaned, deduplicated data

\- \*\*Gold Layer:\*\* Aggregated, business-ready data



\### Component 3: Machine Learning Models

\- \*\*Churn Prediction:\*\* Random Forest (85% accuracy)

\- \*\*LTV Prediction:\*\* Gradient Boosting (R²=0.78)

\- \*\*Recommendations:\*\* Collaborative Filtering (72% precision)



\### Component 4: FastAPI REST API

\- \*\*Port:\*\* 8000

\- \*\*Latency:\*\* <100ms per prediction

\- \*\*Status:\*\* Running 24/7 as Windows Service



\### Component 5: Streamlit Dashboard

\- \*\*Port:\*\* 8501

\- \*\*Features:\*\* Real-time monitoring, interactive predictions, batch processing



\## 🛠️ Tech Stack



| Component | Technology |

|-----------|-----------|

| Data Processing | Databricks, Apache Spark, SQL |

| Orchestration | Apache Airflow 3.2.1 |

| ML Models | Scikit-learn, XGBoost |

| API Server | FastAPI, Uvicorn |

| Dashboard | Streamlit, Plotly |

| Task Scheduling | Windows Task Scheduler |

| Service Management | NSSM (Windows Service) |

| Language | Python 3.12 |



\## 📈 Key Metrics



\- \*\*Daily Data Volume:\*\* 950K+ transactions

\- \*\*Model Accuracy:\*\* 85% (Churn), 0.78 R² (LTV)

\- \*\*API Latency:\*\* <100ms per prediction

\- \*\*Model Training Time:\*\* \~3-5 minutes daily

\- \*\*Service Uptime:\*\* 99.9%



\## 🚀 Deployment



\### Local Setup



1\. \*\*Install Dependencies:\*\*

```bash

pip install -r requirements.txt

```



2\. \*\*Start FastAPI:\*\*

```bash

cd api

python main.py

```



3\. \*\*Start Dashboard:\*\*

```bash

cd dashboard

streamlit run dashboard.py

```



4\. \*\*Access:\*\*

\- API: http://localhost:8000/docs

\- Dashboard: http://localhost:8501



\## 📁 Project Structure

ml-datapipeline-project/

├── airflow/

│   └── customer\_analytics\_dag\_FIXED\_AIRFLOW\_3\_2\_1.py

├── databricks/

│   ├── 01\_extract\_and\_bronze.py

│   ├── 02\_transform\_to\_silver.py

│   └── 03\_create\_gold\_layer.py

├── ml\_models/

│   ├── 03\_train\_models.py

│   ├── 04\_make\_predictions.py

│   └── models/

├── api/

│   └── main.py

├── dashboard/

│   └── dashboard.py

├── docs/

│   ├── ARCHITECTURE.md

│   └── MODEL\_PERFORMANCE.md

├── README.md

├── SETUP.md

├── API\_DOCS.md

├── requirements.txt

└── .gitignore

\## 💼 Interview Talking Points



\*\*30 seconds:\*\* "Built an end-to-end ML pipeline processing 950K+ daily transactions with Databricks ETL, Airflow orchestration, three trained models with 85% accuracy, FastAPI API serving <100ms predictions, and Streamlit dashboard for monitoring."



\*\*2 minutes:\*\* "I architected a production-grade ML data pipeline combining Databricks for scalable ETL, Apache Airflow for orchestration, and three custom-trained models for churn prediction (85% accuracy), lifetime value prediction (R²=0.78), and product recommendations (72% precision). The FastAPI REST API serves predictions in under 100ms, deployed as a Windows Service with auto-restart. A Streamlit dashboard provides real-time monitoring. The entire system runs automatically daily at 2 AM."



\## 🎓 Learning Outcomes



✅ End-to-end ML pipeline architecture

✅ Databricks Delta Lake data warehouse

✅ Apache Airflow orchestration

✅ Spark SQL for distributed computing

✅ Production ML model training

✅ REST API design and deployment

✅ Real-time monitoring dashboards

✅ Windows service management



\## 📞 Contact



\- \*\*GitHub:\*\* https://github.com/Chiku-projects/ml-datapipeline-project





\---



\*\*Built with ❤️ for production-grade ML systems\*\*

=======
# ml-data-pipeline-project
>>>>>>> 155ea7b18357b26ef8c43bf6cb8c8d9dac3d0e44
