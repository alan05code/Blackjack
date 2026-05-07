@echo off
REM Crea e attiva l'ambiente virtuale, installa requisiti, avvia gioco e visione
setlocal

REM Posizionati nella cartella dello script
cd /d "%~dp0"

REM Crea venv se non esiste
if not exist ".venv" (
    python -m venv .venv
)

REM Attiva venv
call ".venv\Scripts\activate.bat"

REM Installa dipendenze di base
pip install --upgrade pip >NUL
pip install -r requirements.txt 2>NUL

REM Avvia run_game e vision_blackjack in due finestre separate
start "run_game" cmd /k ".venv\Scripts\python.exe run_game.py"
start "vision_blackjack" cmd /k ".venv\Scripts\python.exe vision_blackjack.py"

endlocal