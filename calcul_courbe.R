# Charger les bibliothèques nécessaires
library(ggplot2)

# Lire le fichier CSV (remplacez 'votre_fichier.csv' par le chemin réel)
file_path <- "C:/Users/Utilisateur/OneDrive/Bureau/Kevin/sample_400m.csv"  # Remplacez par le chemin de votre fichier

# Lire uniquement les colonnes nécessaires
data <- read.csv(file_path, skip = 3, header = TRUE, colClasses = c(
  "NULL",       # Sample rate (ignorer)
  "character",  # Time (prendre)
  "numeric",    # HR (bpm) (prendre)
  rep("NULL", 8)  # Ignorer les autres colonnes
))

# Renommer les colonnes pour simplifier
colnames(data) <- c("Time", "HR")

# Supprimer les lignes avec des valeurs NA
data <- na.omit(data)

# Convertir la colonne "Time" en objet POSIXct pour faciliter les calculs
data$Time <- as.POSIXct(data$Time, format = "%H:%M:%S")

# Fonction pour calculer une moyenne mobile
moving_average <- function(x, n = 15) { # n = taille de la fenêtre
  stats::filter(x, rep(1/n, n), sides = 2)
}

# Appliquer la moyenne mobile sur les fréquences cardiaques
data$HR_smoothed <- moving_average(data$HR, n = 15)

# Calculer la dérivée de la courbe lissée
# Différence des valeurs lissées divisée par la différence de temps (en secondes)
data$Time_diff <- c(NA, diff(as.numeric(data$Time)))  # Différence en secondes
data$HR_derivative <- c(NA, diff(data$HR_smoothed) / data$Time_diff[-1])  # Dérivée numérique
data$HR_derivative <- moving_average(data$HR_derivative, n=10)

# Vérifier les données calculées
head(data)

# Graphique 1 : Courbe brute et lissée
plot1 <- ggplot(data, aes(x = Time)) +
  geom_line(aes(y = HR), color = "blue", alpha = 0.5, linetype = "dashed", size = 0.8) + # Courbe brute
  geom_line(aes(y = HR_smoothed), color = "red", size = 1) + # Courbe lissée
  labs(
    title = "Fréquence Cardiaque (Brute et Lissée)",
    x = "Temps",
    y = "Fréquence Cardiaque (bpm)"
  ) +
  theme_minimal()

# Graphique 2 : Courbe de la dérivée
plot2 <- ggplot(data, aes(x = Time)) +
  geom_line(aes(y = HR_derivative), color = "green", size = 1) +
  labs(
    title = "Dérivée de la Fréquence Cardiaque",
    x = "Temps",
    y = "Variation de la Fréquence (bpm/s)"
  ) +
  theme_minimal()

# Afficher les graphiques
library(gridExtra)
grid.arrange(plot1, plot2, ncol = 1)
