@echo off
set VENV_DIR=.venv

echo [1/3] Tao moi virtual environment...
python -m venv %VENV_DIR%

echo [2/3] Kich hoat virtual environment va nang cap pip...
call %VENV_DIR%\Scripts\activate
python.exe -m pip install --upgrade pip

echo [3/3] Dang cai dat cac thu vien tu requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
    echo [+] Da cai dat xong tat ca thu vien.
) else (
    echo [X] Loi: Khong tim thay file requirements.txt
)

echo.
echo =========================================
echo    DA HOAN THANH THIET LAP MOI TRUONG
echo =========================================
pause
