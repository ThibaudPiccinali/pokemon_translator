import os
import cv2
import card_identifier as card
import utils as utils
import numpy as np

name_input_folder = 'input_images'
name_output_folder = 'output_images'

img_format = ['.png', '.jpg', '.jpeg']

names = []

reader = card.create_reader(['la'])

# Parcourir tous les fichiers du dossier
for fichier in os.listdir(name_input_folder):
    # Obtenir l'extension du fichier
    extension = os.path.splitext(fichier)[1].lower()

    # Vérifier si le fichier est une image
    if extension in img_format:
        # Ajouter le chemin complet du fichier à la liste
        names.append(fichier)

for name in names:

    image = cv2.imread(f'{name_input_folder}\{name}')

    # Redimensionner l'image
    image_resized = cv2.resize(image, (900, 1200), interpolation = cv2.INTER_AREA)
        
    corners = card.find_good_contours(image_resized)

    if len(corners) == 0:
        cv2.imshow(f'{name_output_folder}\{name}_no_card_found', image_resized)

    else:
       
        # Dessine les contours
        image_contours = utils.draw_rectangle_with_corners(image_resized,corners)        
        cv2.imshow(f'{name}_contours', image_contours)
        
        # On calcule la matrice
        
        # Pour cela on remet d'abord à l'echelle nos corners
        
        corners_well_shaped = []
        for corner in corners:
            corner = [[int(corner[0][0]*10/3), int(corner[0][1]*10/3)]] # le 10/3 c'est pour passer de 900 à 3000 et 1200 à 4000 (pour retrouver la taille de l'image originale)
            corners_well_shaped.append(corner)
        
        corners_well_shaped = np.array(corners_well_shaped)
        
        corners_ordered = utils.reorderCorners([corners_well_shaped[0][0],corners_well_shaped[1][0],corners_well_shaped[2][0],corners_well_shaped[3][0]])  # Reorders corners to [topLeft, topRight, bottomLeft, bottomRight]
                
        pts1 = np.float32(corners_ordered)
                
        # Calcul de la taille de la carte sur l'image
                
        largeur = int(utils.distance(corners_ordered[0][0],corners_ordered[1][0])) # Peut être que ça devrait tenir compte des proportion réelles d'une carte ? genre (330,440)
        longueur = int(utils.distance(corners_ordered[0][0],corners_ordered[2][0]))
                
        pts2 = np.float32([[0, 0], [largeur, 0], [0, longueur], [largeur, longueur]])
        # Makes a matrix that transforms the detected card to a vertical rectangle
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # Extrait la carte
        # Transforms card to a rectangle widthCard x heightCard
        imgWarpColored = cv2.warpPerspective(image,  matrix, (largeur, longueur))
        cv2.imshow(f'{name}_card_detected', imgWarpColored)
        cv2.imwrite(f'{name_output_folder}\{name}_card_detected.png', imgWarpColored)

        # On prend uniquement la partie basse de la carte pour son identification
        nb_segment = 4
        segment_height = longueur // nb_segment
        
        startY = (nb_segment-1) * segment_height
        endY = nb_segment * segment_height
        segment = imgWarpColored[startY:endY, 0:largeur]
        
        print(card.identify_card(segment,reader))
        
        # cv2.imwrite(f'{name_output_folder}\{name}_test_decoupe.jpg', segment)
        
        
cv2.waitKey(0)
cv2.destroyAllWindows()
