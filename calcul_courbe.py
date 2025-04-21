import csv
import os
import re
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

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

# ------------------------------------------
# Chargement et pré-traitement des données
# ------------------------------------------

input_filename = 'LEGOFF_Guillaume_post_test_protocole_400m.csv'

# Extraction des métadonnées depuis le nom de fichier
basename = os.path.basename(input_filename).lower().replace(".csv", "")
parts = re.split(r"[_\-]", basename)

nom = parts[0].upper()
prenom = parts[1].capitalize()
prepost = next((p for p in parts if p in ['pre', 'post']), 'NA')
distance = next((d for d in parts if d.endswith('m')), 'NA').replace('m', '')

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

start_idx = np.argmax(hr_derive) - 30
time_crop = [t - data_time[start_idx] for t in data_time[start_idx:]]
hr_crop = hr_smooth[start_idx:]

# Analyse points clés
idx_fcmax = np.argmax(data_hr)
t_fcmax = time_crop[idx_fcmax - start_idx]
fcmax = int(data_hr[idx_fcmax])

tknee = find_elbow_point(np.array(time_crop[:150]), hr_derive[start_idx:start_idx+150])
fc_knee = int(hr_crop[tknee])
angle = round(math.degrees(math.atan((fcmax - fc_knee) / (t_fcmax - tknee))), 2)

recovery_idx = detect_recovery_start(hr_crop)
t_recovery = time_crop[recovery_idx]
fc_recovery = int(hr_crop[recovery_idx])
fc_180 = int(hr_crop[recovery_idx + 180]) if recovery_idx + 180 < len(hr_crop) else 'NA'
t_180 = time_crop[recovery_idx + 180] if recovery_idx + 180 < len(time_crop) else 'NA'

# Sauvegarde CSV résumé
output_csv = "resume_fc.csv"
write_header = not os.path.exists(output_csv)

with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["Nom", "Prénom", "Pre/Post", "Distance", "FC_kneepoint", "t_kneepoint", "FC_max", "Angle_derive", "FC_180s"])
    writer.writerow([nom, prenom, prepost, distance, fc_knee, tknee, fcmax, angle, fc_180])

# Affichage simple de la FC
print("Knee point", tknee, "secondes")
print("FC max", fcmax, "bpm")
print("Angle derive", angle, "°")
print("FC fin+180s", fc_180, "bpm")

plt.figure(figsize=(10, 5))
plt.plot(time_crop, hr_crop, label="FC lissée", color='black')
plt.axvline(x=tknee, color='blue', linestyle='--', label="Knee Point")
plt.axvline(x=t_fcmax, color='orange', linestyle='--', label="FC Max")
plt.axvline(x=t_recovery, color='green', linestyle='--', label="Début récup")

plt.title("Évolution de la fréquence cardiaque")
plt.xlabel("Temps (s)")
plt.ylabel("FC (bpm)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("courbe_FC.png")
plt.show()