import cv2
import easyocr
import utils as utils
import tcgdex_api as tcg

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

# To compare two cards (and know if they are similars)

# def average_hash(image, hash_size=8):
#     """Calcule l'average hash de l'image."""
#     # Convertir l'image en niveaux de gris
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Redimensionner l'image
#     resized_image = cv2.resize(gray_image, (hash_size, hash_size), interpolation=cv2.INTER_LANCZOS4)
#     # Calculer la moyenne des pixels
#     avg = resized_image.mean()
#     # Générer le hash : chaque bit est 1 si le pixel est supérieur à la moyenne, 0 sinon
#     hash_array = resized_image > avg
#     # Convertir le tableau en hash (sous forme de chaîne de caractères)
#     hash_str = ''.join(hash_array.flatten().astype(int).astype(str))
#     return hash_str

# def hamming_distance(hash1, hash2):
#     """Calcule la distance de Hamming entre deux hashes."""
#     if len(hash1) != len(hash2):
#         raise ValueError("Les hashes doivent avoir la même longueur")
#     return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# def calculate_similarity(image1_cv2, image2_cv2):
#     """Calcule la similarité entre deux images en utilisant les average hashes."""
#     # Pour l'instant n'est pas tres convaincant
    
#     # Calculer les average hashes
#     hash1 = average_hash(image1_cv2)
#     hash2 = average_hash(image2_cv2)
    
#     # Calculer la distance de Hamming entre les deux hashes
#     distance = hamming_distance(hash1, hash2)
    
#     # Calculer la similarité en pourcentage
#     similarity = 1 - (distance / len(hash1))
#     return similarity

def load_image(image_path):
    """Charge l'image depuis le chemin spécifié."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def detect_and_compute_keypoints(image):
    """Détecte les keypoints et calcule les descripteurs avec SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    """Effectue le matching entre les descripteurs des deux images."""
    # Utilisation de FLANN pour matcher les descripteurs
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Retenir les bons matches selon la méthode du ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return good_matches

def calculate_similarity(image1, image2):
    """Calcule la similarité entre deux images en utilisant les keypoints."""
    # Charger les images

    # Détecter les keypoints et les descripteurs
    kp1, des1 = detect_and_compute_keypoints(image1)
    kp2, des2 = detect_and_compute_keypoints(image2)

    # Matcher les descripteurs
    good_matches = match_keypoints(des1, des2)
    
    # Calculer une métrique de similarité basée sur les bons matches
    similarity = len(good_matches) / min(len(kp1), len(kp2))
    return similarity

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
        similarity = calculate_similarity(card_detected,card_image_bdd)
        
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