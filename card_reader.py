import cv2
import easyocr
import utils as utils
import tcgdex_api as tcg

# To extract the code of the card

def create_reader(language):
    return easyocr.Reader(language)

def identify_card_hp(image_cv2,reader):
    
    # On prend la partie haute de la carte pour récupérer les PV
    
    longueur,largeur, _ = image_cv2.shape
    nb_segment = 3
    segment_height = longueur // nb_segment
        
    startY = 0
    endY = segment_height
    top_card = image_cv2[startY:endY, largeur//2:largeur]
    
    filtered_results = []
    
    results = reader.readtext(top_card)
    results_confident = []

    for result in results:
        _, largeur, _ = top_card.shape
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
            x, _ = coord
            if x < largeur//3:
                good_coordinates = False
        if good_coordinates:
            filtered_results.append(result)
    
    if len(filtered_results) < 1:
        return None
    
    for filtered_result in filtered_results:
        try:
            # potentiellement le lecteur a detecté le "hp", on regarde si c'est le cas
            # si ce n'est pas le cas cette étape va planter
            # sinon on retourne le prochain truc détecté par le lecteur (ce qui correspond au hp)
            hp = int(filtered_result[-2])
            return hp
        except:
            hp = None
    
    return None
    
def identify_card_set_and_nb(image_cv2, reader):
    
    # On prend la partie basse de la carte pour récupérer le set et le numéro de la carte
    
    longueur,largeur, _ = image_cv2.shape
    nb_segment = 4
    segment_height = longueur // nb_segment
        
    startY = (nb_segment-1) * segment_height
    endY = nb_segment * segment_height
    bottom_card = image_cv2[startY:endY, 0:largeur]
        
    filtered_results = []
    
    results = reader.readtext(bottom_card)
    results_confident = []
        
    for result in results:
        hauteur, largeur, _ = bottom_card.shape
        if result[-1] > 0.3:
            # On filtre suivant le confident level
            results_confident.append(result) 
    
    if results_confident == []:
        return None,None
        
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
        
    if len(filtered_results) < 2:
        return None,None
    
    # Doit être les deux derniers résultats
    filtered_results = filtered_results[-2:]

    # On extrait uniquement la chaine de caractère
    
    if '/' in filtered_results[1][1]:
        filtered_results = [filtered_results[0][1].lower(),[filtered_results[1][1].split('/')[0],filtered_results[1][1].split('/')[1][:3]]]
    else:
        filtered_results = [filtered_results[0][1].lower(),None]
    
    return filtered_results


import time

if __name__ == '__main__':
    
    start_time = time.time()
    
    card_detected = cv2.imread("output_images/IMG_20240821_160259.jpg_card_detected.png")
    # card_detected = cv2.imread("output_images/test.jpg_card_detected.png")
    best_similarity = 0
    best_card_match = None
    
    set_name = "sv06"
    set_info = tcg.get_set(set_name)
    
    cards_filtered = tcg.filter_card_HP("sv06",60)
    # cards_filtered = tcg.filter_card_category(set_name,"Trainer")
    
    cards_filtered = tcg.get_non_secret_card(set_info,cards_filtered)
    
    for card in cards_filtered:
        card_image_bdd = tcg.get_image_cv2(card)
        similarity = utils.calculate_similarity(card_detected,card_image_bdd)
        
        if similarity> best_similarity:
            best_card_match = card_image_bdd
            best_similarity = similarity
            
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Temps d'exécution: {execution_time:.4f} secondes")
    
    if best_card_match is not None:
        # Afficher l'image en utilisant OpenCV (optionnel)
        cv2.imshow('best_card_match', best_card_match)
        cv2.waitKey(0)
        cv2.destroyAllWindows()