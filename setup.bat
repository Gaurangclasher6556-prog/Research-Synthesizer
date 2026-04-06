@echo off
echo.
echo   ╔══════════════════════════════════════════╗
echo   ║   Research Synthesizer - Setup           ║
echo   ║   Cloud-Deployable Edition (Groq)        ║
echo   ╚══════════════════════════════════════════╝
echo.

:: Create virtual environment
if not exist "venv" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/5] Virtual environment exists.
)

:: Activate
echo [2/5] Activating environment...
call venv\Scripts\activate.bat

:: Install in stages to avoid resolver conflicts
echo [3/5] Installing core dependencies...
pip install --quiet numpy scikit-learn pydantic python-dotenv rich markdown PyMuPDF arxiv

echo [4/5] Installing AI framework...
pip install --quiet streamlit chromadb sentence-transformers

echo [5/5] Installing CrewAI...
pip install --quiet crewai[tools] crewai-tools

echo.
echo   ╔══════════════════════════════════════════╗
echo   ║   Setup Complete!                        ║
echo   ╚══════════════════════════════════════════╝
echo.
echo   Get your FREE Groq API key:
echo     https://console.groq.com
echo.
echo   Launch the app:
echo     run.bat (or: streamlit run app.py)
echo.
echo   Opens at: http://localhost:8501
echo.
pause
