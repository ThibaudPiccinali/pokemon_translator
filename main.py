import cv2
import numpy as np
import tcgdex_api as tcg
import card_reader as card_reader
import utils as utils

# URL du flux vidéo IP Webcam. Assurez-vous que le téléphone et l'ordinateur sont sur le même réseau.
url = "http://192.168.0.109:8080/video"

name_output_folder = 'output_images'
name_table_convertion = 'table_jap_occidental.txt'

reader = card_reader.create_reader(['la'])

# Ouvrir le flux vidéo
cap = cv2.VideoCapture(url)

ret, frame = cap.read()
    
if not ret:
    print("Erreur lors de la lecture du flux vidéo.")
       
y,x,_ = frame.shape
    
size_box_x = 63 # Proportion carte
size_box_y = 88 # Proportion carte
    
while size_box_x + 3*63 < x and size_box_y + 3*88 < y:
    size_box_x +=63
    size_box_y +=88
    
# Définir les coordonnées du rectangle (coin supérieur gauche et coin inférieur droit)
top_left = ((x-size_box_x)//2, (y-size_box_y)//2)  # Coordonnées du coin supérieur gauche
top_right = (x - (x-size_box_x)//2, (y-size_box_y)//2)
bottom_left = ((x-size_box_x)//2, y - (y-size_box_y)//2)
bottom_right = (x - (x-size_box_x)//2, y - (y-size_box_y)//2)  # Coordonnées du coin inférieur droit

# Définir la couleur du rectangle (rouge en BGR)
color = (0, 0, 255)  # BGR pour rouge

# Définir l'épaisseur du rectangle
thickness = 2  # Épaisseur du contour du rectangle

corners_ordered = [[top_left],[top_right],[bottom_left],[bottom_right]]

pts1 = np.float32(corners_ordered)
                
# Calcul de la taille de la carte sur l'image
                    
largeur = int(utils.distance(corners_ordered[0][0],corners_ordered[1][0])) # Peut être que ça devrait tenir compte des proportion réelles d'une carte ? genre (330,440)
longueur = int(utils.distance(corners_ordered[0][0],corners_ordered[2][0]))
                    
pts2 = np.float32([[0, 0], [largeur, 0], [0, longueur], [largeur, longueur]])
# Makes a matrix that transforms the detected card to a vertical rectangle
matrix = cv2.getPerspectiveTransform(pts1, pts2)

while True:
    
    ret, frame = cap.read()
    image = frame.copy()

    # Dessiner le rectangle sur l'image
    
    frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    resized_frame = cv2.resize(frame, (600, 800))
    cv2.imshow('frame', resized_frame)

    # On lance le traitement si 'p' est pressé
    if cv2.waitKey(1) & 0xFF == ord('p'):
            
            # Extrait la carte
            # Transforms card to a rectangle widthCard x heightCard
            imgWarpColored = cv2.warpPerspective(image,  matrix, (largeur, longueur))
                        
            set_name, number_card = card_reader.identify_card_set_and_nb(imgWarpColored,reader)
            hp = card_reader.identify_card_hp(imgWarpColored,reader)
            
            print(set_name, number_card,hp)
              
            if set_name != None and number_card!=None:
                
                # On modifie juste un cas classique (le S confondu avec 5)
                if set_name[0]=="5":
                    set_name="s"+set_name[1:]
                
                # On convertit le nom de set japonais en nom de set occidental
                
                convert_table = utils.lire_table_fichier(name_table_convertion)
                set_name_translate = utils.obtenir_valeur(convert_table, set_name)
                
                if set_name_translate!=None:
                
                    set_info = tcg.get_set(set_name_translate)
                    
                    best_similarity = 0
                    best_card_match = None
                    
                    if hp == None:
                        # Alors ce n'est pas un pokemon
                        cards_filtered = tcg.filter_card_category(set_name_translate,"Non Pokemon")
                    else:
                        cards_filtered = tcg.filter_card_HP(set_name_translate,hp)
                    
                    if int(number_card[0]) > int(number_card[1]):
                        # Alors c'est une secrete
                        cards_filtered = tcg.get_secret_card(set_info,cards_filtered)
                    else:
                        # Ce n'est pas une secrete
                        cards_filtered = tcg.get_non_secret_card(set_info,cards_filtered)
                            
                    for card in cards_filtered:
                        card_image_bdd = tcg.get_image_cv2(card)
                        similarity = utils.calculate_similarity(imgWarpColored,card_image_bdd)
                        
                        if similarity> best_similarity:
                            best_card_match = card_image_bdd
                            best_similarity = similarity
                            
                    # Afficher l'image finale'
                    cv2.imshow('Card match', best_card_match)
            
    # Sortir de la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
