@echo off
REM ML Pipeline Scheduler - Runs daily at 2 AM
REM File: C:\Users\Chirag\Downloads\airflow_project\run_pipeline.bat

cd C:\Users\Chirag\Downloads\airflow_project

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run pipeline (full - Components 2 & 3)
python run_pipeline.py --mode run-now

REM Pause so you can see results
pause