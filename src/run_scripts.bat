@echo off
REM Navigate to the directory of the .bat file
cd /d %~dp0

REM Navigate to the project directory (one level up from src)
cd ..

REM Activate the virtual environment
call .venv\Scripts\activate

REM Navigate back to the src directory
cd src

REM Run the preprocessing script
python preprocessing.py

REM Run the Streamlit app
streamlit run main.py

REM Pause to keep the terminal open after execution
pause
