import cv2
import easyocr
import utils as utils

# To find the card

def find_good_contours(image_cv2):
    
    # gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(image_cv2, (1, 1), 0)

    # Détecter les bords avec Canny
    edges = cv2.Canny(blurred, 50, 200)

    # Trouver les contours dans l'image
    contours, _  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours et détecter les coins
    for contour in contours:
        
        # Approximer les contours en un polygone
        epsilon = 0.02 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)
        
        # Si le polygone a 4 côtés, ça peut être un paralelogramme
        if len(corners) == 4:
            # On teste si c'est un paralelogramme
            if(utils.is_paralelogramme([corners[0][0],corners[1][0],corners[2][0],corners[3][0]])):
                # On considère que il ne peut y avoir qu'une carte par image donc la première qu'on trouve doit petre la bonne
                return corners
    
    return []

# To extract the code of the card

def create_reader(language):
    return easyocr.Reader(language)

def identify_card(image_cv2, reader):
    filtered_results = []
    results = reader.readtext(image_cv2)
    # On utilise les coordonnées des résultat pour déterminer qui est qui
    for result in results:
       
        if result[-1] > 0.4:
            # On filtre suivant le confident level
            filtered_results.append(result) 
    
    return filtered_results
