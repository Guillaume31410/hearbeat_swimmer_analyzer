#!/bin/bash

SOURCE_DIR="SOURCE_CSV"
SCRIPT="calcul_courbe.py"

# Vérifier si le dossier existe
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Le dossier $SOURCE_DIR n'existe pas."
  exit 1
fi

# Récupérer tous les fichiers CSV
CSV_FILES=("$SOURCE_DIR"/*.csv)
COUNT=${#CSV_FILES[@]}

# Vérifier qu'il y a des fichiers
if [ "$COUNT" -eq 0 ]; then
  echo "Aucun fichier CSV dans $SOURCE_DIR."
  exit 1
fi

# Boucle sur les fichiers
CURRENT=1
for FILE in "${CSV_FILES[@]}"; do
  echo "Traitement du fichier $CURRENT/$COUNT : $(basename "$FILE")"
  python3 "$SCRIPT" "$FILE"
  ((CURRENT++))
done

echo "Tache terminee !"
