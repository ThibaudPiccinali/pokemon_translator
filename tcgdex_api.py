import requests
import numpy as np
import cv2

# Configuration de l'URL de base de l'API
base_url = "https://api.tcgdex.net/v2/en"

def get_set(set_name):
    url = f"{base_url}/sets/{set_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur {response.status_code}: {response.text}")
        return None

def get_cards(set_name):
    set = get_set(set_name)
    return set['cards']

def get_image_cv2(card):
    card_image_url = f'{card["image"]}/high.webp'
    try:
        # Envoyer une requête GET à l'URL
        response = requests.get(card_image_url)
        # Vérifier que la requête a réussi
        response.raise_for_status()
        
        # Convertir les données de l'image en un tableau NumPy
        image_data = np.frombuffer(response.content, np.uint8)
        # Décoder l'image à l'aide de OpenCV
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is not None:
            return image
        else:
            print("Erreur lors du décodage de l'image.")
            return None
    except requests.RequestException as e:
        print(f"Erreur lors du téléchargement de l'image : {e}")
        return None
    
if __name__ == '__main__':
    
    image = get_image_cv2(get_cards("sv01")[-1])
    
    if image is not None:
        # Afficher l'image en utilisant OpenCV (optionnel)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    