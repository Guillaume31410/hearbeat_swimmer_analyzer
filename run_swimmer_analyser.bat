@echo off
setlocal ENABLEDELAYEDEXPANSION

set SOURCE_DIR=SOURCE_CSV
set SCRIPT=calcul_courbe.py

rem Initialiser le compteur total de fichiers
set /a COUNT=0

rem Compter le nombre total de fichiers CSV dans SOURCE_CSV
for %%f in ("%SOURCE_DIR%\*.csv") do (
    set /a COUNT+=1
)

rem Si aucun fichier n'est trouvé
if "!COUNT!"=="0" (
    echo Aucun fichier CSV dans %SOURCE_DIR%.
    pause
    exit /b
)

rem Initialiser le compteur pour la progression à 1
set /a CURRENT=1

rem Parcourir tous les fichiers CSV dans SOURCE_CSV
for %%f in ("%SOURCE_DIR%\*.csv") do (
    echo Traitement du fichier !CURRENT!/!COUNT! : %%~nxf
    python "%SCRIPT%" "%%f"
    set /a CURRENT+=1
)

echo Tache terminee !
pause
