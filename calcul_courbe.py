import csv
from turtle import color
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime
import numpy as np

###############################################################################
#                        FUNCTION DEFINITION
###############################################################################

def butterworth_filter(data, cutoff, fs, order=5, filter_type='low'):
    """
    Applique un filtre de Butterworth sur un tableau de données.
    
    :param data: Le tableau de données à filtrer (numpy array).
    :param cutoff: La fréquence de coupure du filtre (Hz).
    :param fs: La fréquence d'échantillonnage du signal (Hz).
    :param order: L'ordre du filtre Butterworth (défaut: 5).
    :param filter_type: Type de filtre ('low' pour passe-bas, 'high' pour passe-haut, 
                        'bandpass' pour passe-bande, 'bandstop' pour coupe-bande).
    :return: Le tableau filtré.
    """
    nyquist = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = np.array(cutoff) / nyquist  # Normalisation de la fréquence de coupure

    # Conception du filtre
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)

    # Application du filtre avec filtfilt (évite le déphasage)
    filtered_data = filtfilt(b, a, data)

    return filtered_data.tolist()  # Conversion en liste

#----------------------------------
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
#----------------------------------

#----------------------------------
def dichotomie(data1, data2):
    incr = 0
    data = data1 - data2
    borne_inf = 0
    borne_sup = data1.index(max(data1))
    
    while abs(borne_inf-borne_sup) > 1 :
        if incr > len(data) :
            print("loop error")
            break 
        
        half = (borne_inf+borne_sup)//2
        if data[half] > 0 :
            borne_sup = half
        else :
            borne_inf = half
        
        incr = incr+1

    return borne_inf
 #----------------------------------                  

def tableau_derive(tableau):
    """
    Calcule le tableau dérivé en prenant les différences successives entre 
    les éléments consécutifs d'un tableau donné.

    :param tableau: Liste des nombres
    :return: Liste des différences consécutives
    """
    if len(tableau) < 2:
        # Un tableau avec moins de 2 éléments n'a pas de dérivée
        return []

    derive = []
    for i in range(len(tableau) - 1):
        derive.append(tableau[i + 1] - tableau[i])
    
    derive.append(0)
    return derive 

###############################################################################
#                            MAIN FUNCTION
###############################################################################

data_raw = []
data_time =[]
data_measure = []
data_derive = []

with open('sample_400m.csv', mode='r', encoding='utf-8') as f:
    lecteur = csv.reader(f)
    for index, ligne in enumerate(lecteur):
        if index >= 3:
           data_raw.append(ligne)
           
# Parcourir les données pour remplir les listes
for row in data_raw:
    if row[1] and row[2]:  # Vérifie si les colonnes "temps" et "mesure" sont présentes
        # Conversion de hh:mm:ss en secondes
        time_obj = datetime.strptime(row[1], "%H:%M:%S")
        time_in_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        data_time.append(time_in_seconds)  # Ajoute le temps en secondes
        data_measure.append(float(row[2]))  # Convertit en float pour les mesures    
        
#lissage de la courbe
data_smooth = butterworth_filter(data_measure, 20, 100, order=5, filter_type='low')

#calcul derivee
data_derive = tableau_derive(data_smooth)

#calcul max
x_max_derive = data_derive.index(max(data_derive))
print("1er point d'inflexion (montee en charge) :", data_time[x_max_derive], "secondes")

#calcul min 
x_min_derive = data_derive.index(min(data_derive))
print("2nd point d'inflexion (recuperation) :", data_time[x_min_derive], "secondes")

# affichage des traces
# Création des deux graphiques
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)


# Premier graphique : tableau initial
axs[0].plot(data_time, data_smooth, marker='', color='black', label="Tableau initial", linewidth=2)
axs[0].set_title("Fréquence cardiaque mesurée au capteur OH1", fontsize=16)
axs[0].set_ylabel("FC", fontsize=14)
axs[0].grid(True)
#axs[0].axvline(x=x_max_derive, color="red", linestyle="--")
#axs[0].axvline(x=x_min_derive, color="red", linestyle="--")

# Deuxième graphique : tableau dérivé
axs[1].plot(data_time, data_derive, marker='', color='red', label="Tableau dérivé", linewidth=2)
axs[1].set_xlabel("Temps (s)", fontsize=14)
axs[1].set_ylabel("Derivée de la FC", fontsize=14)
axs[1].grid(True)
#axs[1].axvline(x=x_max_derive, color="red", linestyle="--")
#axs[1].axvline(x=x_min_derive, color="red", linestyle="--")

# Ajustement de l'espacement
plt.tight_layout()

# Sauvegarder avec un fond transparent
plt.savefig("FC_Kevin.png", transparent=True)

plt.show()