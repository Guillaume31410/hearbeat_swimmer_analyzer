import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import butter, filtfilt
from datetime import datetime

###############################################################################
#                        FUNCTION DEFINITION
###############################################################################

def butterworth_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data).tolist()


def tableau_derive(tableau):
    return np.diff(tableau).tolist() + [0]


def find_elbow_point(x, y):
    """Détection du point de coude basée sur la première dérivée et le changement le plus brusque."""
    x, y = np.array(x), np.array(y)
    y_diff = np.diff(y)  # Première dérivée

    # Recherche de l'indice où la chute est la plus forte
    elbow_index = np.argmin(y_diff)  # L'endroit où la pente devient la plus négative

    return x[elbow_index]

def lissage_courbe(data, window_size):
    if window_size % 2 == 0:
        raise ValueError("La taille de la fenêtre doit être impaire.")
        
    smoothed_data = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        # Gestion des bords : on ajuste la fenêtre
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        
        # Moyenne des points dans la fenêtre
        smoothed_data.append(sum(window) / len(window))
    
    return smoothed_data 

# Fonction pour détecter le début de la récupération (descente de la FC)
def detect_recovery_start(data):
    peak_idx = np.argmax(data)  # Index de la FC max
    for i in range(peak_idx + 1, len(data)):
        if data[i] < data[peak_idx]:
            return i  # Retourne l'index du début de la descente
    return None  # Si pas trouvé

###############################################################################
#                            MAIN FUNCTION
###############################################################################

data_time, data_measure = [], []

# Lecture des données depuis le fichier CSV
#with open('Guillaume_pre_test_protocole.csv', mode='r', encoding='utf-8') as f:
with open('sample_400m.csv', mode='r', encoding='utf-8') as f:
    lecteur = csv.reader(f)
    next(lecteur), next(lecteur), next(lecteur)  # Ignorer les trois premières lignes
    
    for row in lecteur:
        if row[1] and row[2]:
            time_obj = datetime.strptime(row[1], "%H:%M:%S")
            data_time.append(time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second)
            data_measure.append(float(row[2]))

# Lissage de la courbe
fs = 100  # Fréquence d'échantillonnage hypothétique
cutoff = 5  # Fréquence de coupure

data_smooth = butterworth_filter(data_measure, cutoff, fs)
data_smooth = lissage_courbe(data_smooth, 29)

data_derive = tableau_derive(data_smooth)

# Suppression du début inutile
start = np.argmax(data_derive)-30
stop = np.argmax(data_smooth)-start
data_crop = data_smooth[start:]
data_derive = tableau_derive(data_crop)
data_derive = lissage_courbe(data_derive, 71)

data_derive_sec = tableau_derive(data_derive)
# on recale le début du graphe sur le début de l'épreuve
data_time = [x - data_time[start] for x in data_time[start:]]

# Application de la méthode du coude pour identifier un point clé
tps_values = np.array(data_time)
fc_values = np.array(data_crop)

#Recherche de la FCmax sur valeur brute
# Trouver l'indice de la valeur maximale de y
indice_FCmax = np.argmax(data_measure)

# Coordonnées du maximum
tFCmax = data_time[indice_FCmax]
FCmax = data_measure[indice_FCmax]

# Coordonnées du knee point
tknee_point = find_elbow_point(tps_values[:150], data_derive[:150])
FC_knee = data_crop[tknee_point]

#calcul angle derive cardiaque
angle = math.atan((FCmax-FC_knee) / (tFCmax - tknee_point)) #arctan((FCmax - FCknee) / (t_FCmax-T_Knee))
angle = np.degrees(angle)

# FC @fin de test +180s
# Debut de la phase de récup ...
recovery_start_idx = detect_recovery_start(data_crop)

# ... FC 180s après
FC_recup = data_crop[recovery_start_idx+180]


# Affichage des résultats
print("\tFCmax :", int(FCmax), "bpm")
print("\tt_Knee Point :", tknee_point, "secondes (Méthode du Coude)")
print("\tangle : ", round(angle, 2), "°")

print("\tFC 180s après arrêt :", int(FC_recup), "bpm")

# Affichage des graphiques
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axs[0].plot(data_time, data_crop, color='black', linewidth=2)
axs[0].set_title("Fréquence cardiaque mesurée au capteur OH1")
axs[0].set_ylabel("FC")
axs[0].grid(True)

# Affichage du knee point
axs[0].axvline(x=tknee_point, color='blue', linestyle='--', label="Knee Point")
axs[0].axvline(x=stop, color='blue', linestyle='--', label="Fc max")

axs[0].legend()

axs[1].plot(data_time, data_derive, color='red', linewidth=2)
axs[1].set_xlabel("Temps (s)")
axs[1].set_ylabel("Dérivée de la FC")
axs[1].grid(True)

plt.tight_layout()
plt.savefig("FC_Kevin.png", transparent=True)
plt.show()
