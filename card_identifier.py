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
    results_confident = []
    
    for result in results:
        hauteur, largeur, _ = image_cv2.shape
        if result[-1] > 0.4:
            # On filtre suivant le confident level
            results_confident.append(result) 
    
    if results_confident == []:
        return None
    
    # On utilise les coordonnées des résultats pour déterminer qui est qui
    for result in results_confident:
        coordinates = result[0]
        good_coordinates = True
        for coord in coordinates:
            x, y = coord
            if (x < 0) or (x > largeur//3):
                good_coordinates = False
            if (y < hauteur//3) or (y > hauteur):
                good_coordinates = False
        if good_coordinates:
            filtered_results.append(result)
    
    if len(results_confident) < 2:
        return None
    
    # Doit être les deux derniers résultats
    filtered_results = filtered_results[-2:]
    
    # On extrait uniquement la chaine de caractère
    
    filtered_results = [filtered_results[0][1],[filtered_results[1][1].split('/')[0],filtered_results[1][1].split('/')[1][:3]]]
    
    return filtered_results
