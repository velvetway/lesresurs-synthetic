@echo off
chcp 65001 >nul
cd /d "%~dp0backend"
where python >nul 2>nul || (echo Не найден Python 3.10+. Установите с python.org и отметьте "Add to PATH". & pause & exit /b 1)
if not exist .venv (
  echo Создаю виртуальное окружение...
  python -m venv .venv || (pause & exit /b 1)
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt || (pause & exit /b 1)
) else (
  call .venv\Scripts\activate.bat
)
echo.
echo ЛесРесурс запущен: http://localhost:8000   (остановить: Ctrl+C)
start "" http://localhost:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
