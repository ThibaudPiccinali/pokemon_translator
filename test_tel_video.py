import cv2

# URL du flux vidéo IP Webcam. Assurez-vous que le téléphone et l'ordinateur sont sur le même réseau.
url = "http://192.168.0.109:8080/video"  # Remplacez cette URL par l'URL donnée par IP Webcam ou une application similaire.

# Ouvrir le flux vidéo
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture du flux vidéo.")
        break
    
    # Afficher le flux vidéo
    cv2.imshow('Flux vidéo en temps réel', frame)
    print(frame.shape)
    # Sortir de la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
