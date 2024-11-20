import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

###############################################################################
#                        FUNCTION DEFINITION
###############################################################################

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

###############################################################################
#                            MAIN FUNCTION
###############################################################################

data_raw = []
data_time =[]
data_measure = []

with open('C:/Users/Utilisateur/OneDrive/Bureau/Kevin/sample_400m.csv', mode='r', encoding='utf-8') as f:
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
data_smooth = lissage_courbe(data_measure, 13)

#calcul temps de montee (0.95% de FCmax)
vector_95FCmax = np.ones(len(data_time))*max(data_measure)*0.95

#calcul d'intersection FC / temps de montée
intersection_point = dichotomie(data_smooth, vector_95FCmax)

#affichage des traces
plt.plot(data_time, data_smooth)
plt.plot(data_time, vector_95FCmax)
plt.axvline(x=intersection_point, color="red")

plt.grid()

# Reduire les ticks sur l'axe x
step = 50  # Espacement entre ticks
plt.xticks(range(min(data_time), max(data_time) + 1, step))  # Ticks réguliers
plt.yticks(range(int(min(data_measure)/10)*10, int((max(data_measure))/10)*10, 10))  # Ticks réguliers pour y (tous les 2)

