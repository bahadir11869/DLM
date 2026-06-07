@echo off
REM ============================================================
REM  DLM Algoritma - DC Sarj Optimizasyonu Dashboard baslatici
REM  Cift tiklayarak calistirin. Tarayicida acilir.
REM ============================================================
cd /d "%~dp0"
echo Dashboard baslatiliyor... (Tarayici otomatik acilir)
echo Durdurmak icin bu pencerede Ctrl+C yapin.
echo.
python -m streamlit run app.py
echo.
echo Dashboard kapandi. Cikmak icin bir tusa basin.
pause >nul
