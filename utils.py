import cv2
import math
import numpy as np

def distance(p1, p2):
    """Calcule la distance entre deux points p1 et p2."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_paralelogramme(points):
    """Détermine si les quatre points forment presque un paralélogramme."""
    
    if len(points) != 4:
        raise ValueError("Il faut exactement quatre points.")
    
    hypothenuseAD = 0
    
    A = 0
    D = 0
    
    for i in range(4):
        for j in range(i+1,4):
            if distance(points[i], points[j]) > hypothenuseAD:
                hypothenuseAD = distance(points[i], points[j])
                A = i
                D = j
    if(hypothenuseAD<250):
        return False
    
    if A == 0:
        if D == 1:
            B = 2
            C = 3
        if D == 2:
            B = 1
            C = 3
        if D == 3:
            B = 2
            C = 1
    if A == 1:
        if D ==2 :
            B = 0
            C = 3
        if D ==3:
            B = 2
            C = 0
    if A == 2:
        if D == 3:
            B = 1
            C = 0
       
    milieu_AD = ((points[A][0]+points[D][0])/2,(points[A][1]+points[D][1])/2)
    milieu_BC = ((points[B][0]+points[C][0])/2,(points[B][1]+points[C][1])/2)
    
    if abs(milieu_AD[0] - milieu_BC[0]) < hypothenuseAD*3/100 and abs(milieu_AD[1] - milieu_BC[1]) < hypothenuseAD*3/100:
        return True
    
    return False

def draw_rectangle_with_corners(image, corners):
    
    image_with_card = image.copy()
    
    # Dessiner le rectangle détecté
    cv2.drawContours(image_with_card, [corners], -1, (0, 255, 0), 2)
                
    # Dessiner les coins
    cv2.circle(image_with_card, tuple(corners[0][0]), 5, (255, 0, 0), -1)
    cv2.circle(image_with_card, tuple(corners[1][0]), 5, (255, 0, 0), -1)
    cv2.circle(image_with_card, tuple(corners[2][0]), 5, (255, 0, 0), -1)
    cv2.circle(image_with_card, tuple(corners[3][0]), 5, (255, 0, 0), -1)
    
    return image_with_card

def card_projection(corners_card,size_card,base_image,card_image):
    
    # Calcul de la taille de la carte sur l'image
    largeur,longueur = size_card 
    pts1 = np.float32(corners_card)   
    pts2 = np.float32([[0, 0], [largeur, 0], [0, longueur], [largeur, longueur]])

    # On resize la taille de l'image carte (pour qu'elle correspondent à ce que l'on a)
    card_image_resized = cv2.resize(card_image, size_card)
    
    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    
    # Obtenir la taille de l'image de base
    base_rows, base_cols, _ = base_image.shape

    # Ajuster la matrice de transformation en fonction de la taille des images
    # Ici, on suppose que la transformation est déjà prévue pour s'appliquer sur l'image overlay.
    transformed_overlay = cv2.warpPerspective(card_image_resized, matrix, (base_cols, base_rows))

    # Créer un masque à partir de l'image overlay transformée
    mask = cv2.cvtColor(transformed_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Inverser le masque pour enlever la partie de l'image de base où l'image overlay sera placée
    mask_inv = cv2.bitwise_not(mask)

    # Enlever la partie correspondante de l'image de base
    base_image_bg = cv2.bitwise_and(base_image, base_image, mask=mask_inv)

    # Prendre seulement la région d'intérêt de l'image overlay
    overlay_fg = cv2.bitwise_and(transformed_overlay, transformed_overlay, mask=mask)

    # Ajouter les deux images
    result = cv2.add(base_image_bg, overlay_fg)

    return result

# Swaps the values of at two indexes in the given array
def swap(arr, ind1, ind2):
    temp = arr[ind1]
    arr[ind1] = arr[ind2]
    arr[ind2] = temp

# Returns sorted array and array of indexes of locations of original values
# Selection sort is used as efficieny won't matter as much for n = 3 or 4
def sortVals(vals):
    indexes = list(range(len(vals)))
    for i in range(len(vals)):
        index = i
        minval = vals[i]
        for j in range(i, len(vals)):
            if vals[j] < minval:
                minval = vals[j]
                index = j
        swap(vals, i, index)
        swap(indexes, i, index)
    return vals, indexes

def reorderCorners(corners):
    # Copy corner values into xvals and yvals
    xvals = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
    yvals = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]

    # Sort yvals and get indexes of original values in sorted array
    yvals, idxs = sortVals(yvals)

    # Change xvals to same order as yvals
    temp = xvals.copy()
    for i in range(len(idxs)):
        xvals[i] = temp[idxs[i]]

    # Check if card is horizontal or vertical and make sure [0, 0] is point closest to top left of image (smallest x)
    if yvals[0] == yvals[1]:
        if xvals[1] < xvals[0]:
            # yvals are same so only swap xvals
            tempx = xvals[0]
            xvals[0] = xvals[1]
            xvals[1] = tempx

    # Find distance from corner with min y to corners
    dist1 = math.sqrt((xvals[1] - xvals[0]) ** 2 + (yvals[1] - yvals[0]) ** 2)
    dist2 = math.sqrt((xvals[2] - xvals[0]) ** 2 + (yvals[2] - yvals[0]) ** 2)
    dist3 = math.sqrt((xvals[3] - xvals[0]) ** 2 + (yvals[3] - yvals[0]) ** 2)
    dists = [dist1, dist2, dist3]

    # Sort distances and get indexes of original values in sorted array
    distSorted, idxsDist = sortVals(dists.copy())

    # Reformat index array to be 4 values, not necessary but makes code easier to read
    idxsDist.insert(0, 0)
    idxsDist[1] += 1
    idxsDist[2] += 1
    idxsDist[3] += 1

    # Check if card is vertical/horizontal
    if yvals[0] == yvals[1]:
        if dists[0] == distSorted[0]:  # If card is vertical; corner [0, 0] is top left of card
            topleft = [xvals[idxsDist[0]], yvals[idxsDist[0]]]  # Same as [xvals[0], yvals[0]]
            topright = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
        else:  # If card is horizontal; corner [0, 0] is top right of card
            topleft = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
    else:  # Else card is tilted in some other orientation
        if xvals[idxsDist[1]] == min(xvals):  # Left-most point is the closest to the point with the smallest y value
            # Left-most point is top left corner
            topleft = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
        else:  # Corner [0, 0] is the top left corner
            topleft = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            topright = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]

    return [[topleft], [topright], [bottomleft], [bottomright]]

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
            if(is_paralelogramme([corners[0][0],corners[1][0],corners[2][0],corners[3][0]])):
                # On considère que il ne peut y avoir qu'une carte par image donc la première qu'on trouve doit petre la bonne
                return corners
    
    return []

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

def lire_table_fichier(nom_fichier):
    table = {}
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()
        for ligne in lignes:
            elements = ligne.split()
            if len(elements) == 2:
                table[elements[0]] = elements[1]
            else:
                print(f"Ligne ignorée car elle ne contient pas exactement deux éléments : {ligne}")
    return table

def obtenir_valeur(table, cle):
    return table.get(cle, None)