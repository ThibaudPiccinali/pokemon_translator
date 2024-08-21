# Qualité de résultat vraiment pas ouf


from PIL import Image
import pytesseract

# Chemin vers l'exécutable de Tesseract (nécessaire sur Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Charger l'image contenant les caractères
image_path = 'test_cropped2.png'
image = Image.open(image_path)

# Utiliser Tesseract pour extraire le texte de l'image
extracted_text = pytesseract.image_to_string(image, config='--psm 6')

# Afficher le texte extrait
print("Texte extrait :")
print(extracted_text)
