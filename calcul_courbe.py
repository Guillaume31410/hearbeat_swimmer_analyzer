import csv
import os
import re
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from datetime import datetime

# ------------------------------------------
# Fonctions utilitaires
# ------------------------------------------

def butterworth_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data).tolist()

def tableau_derive(data):
    return np.diff(data).tolist() + [0]

def lissage_courbe(data, window_size):
    if window_size % 2 == 0:
        raise ValueError("La taille de la fenêtre doit être impaire.")
    return [np.mean(data[max(0, i - window_size//2): min(len(data), i + window_size//2 + 1)]) for i in range(len(data))]

def find_elbow_point(x, y):
    return x[np.argmin(np.diff(y))]

def detect_recovery_start(data):
    peak_idx = np.argmax(data)
    for i in range(peak_idx + 1, len(data)):
        if data[i] < data[peak_idx]:
            return i
    return None

def detect_effort_start(data):
    """
    Détecte le début de l'effort en trouvant le premier point où la FC augmente nettement.
    Retourne l'index correspondant.
    """
    baseline = np.median(data[:30])  # moyenne/valeur de repos au début
    seuil = baseline + 5  # seuil arbitraire (ex: +5 bpm au-dessus du repos)

    for i in range(30, len(data)):
        if data[i] > seuil:
            return i
    return None

def detect_effort_start_by_derivative(data, threshold=0.5, window=5):
    """
    Détecte le début de l'effort en cherchant la première pente (dérivée) suffisamment forte.
    
    :param data: liste des FC (lissées)
    :param threshold: pente minimale indicative pour considérer que la montée commence
    :param window: taille de la fenêtre moyenne pour lisser la dérivée
    :return: index du début de montée ou None
    """
    deriv = tableau_derive(data)
    deriv_smooth = lissage_courbe(deriv, window)

    for i in range(len(deriv_smooth)):
        if deriv_smooth[i] >= threshold:
            return i
    return None

def find_point_after_recovery_at_bpm(target_bpm, hr_data, time_data, recovery_idx):
    """
    Trouve le premier point où la FC atteint la valeur 'target_bpm' après le point de récupération.
    
    :param target_bpm: (float/int) la valeur de FC recherchée (ex: 130 bpm)
    :param hr_data: (list) liste des valeurs de fréquence cardiaque lissées
    :param time_data: (list) liste des timestamps correspondants
    :param recovery_idx: (int) index du point de recovery dans les listes
    :return: (time, bpm) tuple ou None si non trouvé
    """
    for i in range(recovery_idx, len(hr_data)):
        if hr_data[i] <= target_bpm:
            return time_data[i], hr_data[i]
    return None


# ------------------------------------------
# Entrée par argument de ligne de commande
# ------------------------------------------
#for test : LEGOFF_Guillaume_post_test_protocole_400m.CSV
if len(sys.argv) < 2:
    print("Usage: python script.py <nom_du_fichier.csv>")
    sys.exit(1)

input_filename = sys.argv[1]

# Extraction des métadonnées depuis le nom de fichier
basename = os.path.splitext(os.path.basename(input_filename))[0]
parts = re.split(r"[_\-]", basename)

nom = parts[0]  # Première partie du nom, avec une majuscule au début
prenom = parts[1]  # Deuxième partie du nom, avec une majuscule au début
distance = parts[2]  # La distance est directement la troisième partie
prepost = parts[3]  # La quatrième partie correspond à pre/post, en minuscule

# Lecture du fichier CSV
data_time, data_hr = [], []
with open(input_filename, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for _ in range(3): next(reader)
    for row in reader:
        if row[1] and row[2]:
            t = sum(x * int(v) for x, v in zip([3600, 60, 1], row[1].split(":")))
            data_time.append(t)
            data_hr.append(float(row[2]))

# Traitement du signal
fs, cutoff = 100, 5
hr_smooth = lissage_courbe(butterworth_filter(data_hr, cutoff, fs), 29)
hr_derive = tableau_derive(hr_smooth)


OFFSET = 60
start_idx = np.argmax(hr_derive) - OFFSET #derivee maximale
t_derive_max = start_idx + OFFSET
fc_derive_max = int(hr_derive[start_idx])
time_crop = [t - data_time[start_idx] for t in data_time[start_idx:]]
hr_crop = hr_smooth[start_idx:]

effort_start_idx = detect_effort_start_by_derivative(hr_crop)
if effort_start_idx is not None:
    t_effort_start = time_crop[effort_start_idx]
    fc_effort_start = int(hr_crop[effort_start_idx])

############ Analyse points clés ############
# Coordonées FC max
idx_fcmax = np.argmax(data_hr)
t_fcmax = time_crop[idx_fcmax - start_idx]
fcmax = int(data_hr[idx_fcmax])

# Coordonnes Knee point
tknee = find_elbow_point(np.array(time_crop[:150]), hr_derive[start_idx:start_idx+150])
fc_knee = int(hr_crop[tknee])

# Calcul composante rapide
t_compo_rapide = tknee - t_effort_start

# Coordonnees "point de recuperation"
recovery_idx = detect_recovery_start(hr_crop)
t_recovery = time_crop[recovery_idx]
fc_recovery = int(hr_crop[recovery_idx])

# Angle knnepoint - fin d'epreuve
angle = round(math.degrees(math.atan((fc_recovery - fc_knee) / (t_recovery - tknee))), 2)

# Coordonnees 180s after debut de recup
fc_180 = int(hr_crop[recovery_idx + 180]) if recovery_idx + 180 < len(hr_crop) else 'NA'
t_180 = time_crop[recovery_idx + 180] if recovery_idx + 180 < len(time_crop) else 'NA'

# Coordonnée 130 bpm (après point de récupération)
result_130 = find_point_after_recovery_at_bpm(130, hr_crop, time_crop, recovery_idx)
if result_130:
    t_130, fc_130 = result_130
    angle_recup = round(math.degrees(math.atan((t_130-t_recovery) / (fc_recovery - fc_130))), 2)
else:
    t_130, fc_130, angle_recup = 'NA', 'NA', 'NA'


############ Sauvegarde ############
# Sauvegarde CSV résumé
output_dir = "OUTPUT"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Crée le dossier OUTPUT

now = datetime.now() #date et heure actuelle
date_time = now.strftime("%Y%m%d_%H%M%S")  # Exemple : 20250501_075432
output_csv = os.path.join(output_dir, f"resume_fc_{date_time}.csv")
write_header = not os.path.exists(output_csv)

with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
                        "Nom", 
                        "Prenom", 
                        "Pre/Post", 
                        "Distance",
                        "FC start", 
                        "FC kneepoint",
                        "Duree composante rapide",
                        "Angle derive cardiaque",
                        "Angle recuperation",
                        "FC apres 180s"])
    writer.writerow([
                    nom, 
                    prenom, 
                    prepost, 
                    distance, 
                    fc_effort_start, 
                    fc_knee, 
                    t_compo_rapide,
                    angle, 
                    angle_recup, 
                    fc_180])
    
# Affichage simple de la FC
plt.figure(figsize=(10, 5))
plt.plot(time_crop, hr_crop, label="FC lissée", color='black')
plt.axvline(x=tknee, color='blue', linestyle='--', label="Knee Point")
plt.axvline(x=t_recovery, color='green', linestyle='--', label="Début récup")
plt.axvline(x=t_effort_start, color='orange', linestyle='--', label="Début effort")

plt.title("Évolution de la fréquence cardiaque")
plt.xlabel("Temps (s)")
plt.ylabel("FC (bpm)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Création du dossier output s'il n'existe pas
output_dir = "OUTPUT"
os.makedirs(output_dir, exist_ok=True)

# Nom du fichier image basé sur le nom du fichier CSV
image_filename = os.path.splitext(os.path.basename(input_filename))[0] + ".png"
image_path = os.path.join(output_dir, image_filename)
plt.savefig(image_path)

#plt.show()