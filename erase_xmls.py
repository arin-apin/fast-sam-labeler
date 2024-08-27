import os

# Directorios

image_dir = '/home/pablo/DS/cobopa/OD/25082024/images'
annotation_dir = '/home/pablo/DS/cobopa/OD/25082024/Annotations'
# Lista de nombres de imágenes (sin extensión)
image_files = {os.path.splitext(filename)[0] for filename in os.listdir(image_dir) if filename.endswith('.png')}

# Recorrer archivos en el directorio de anotaciones
for annotation_file in os.listdir(annotation_dir):
    if annotation_file.endswith('.xml'):
        annotation_name = os.path.splitext(annotation_file)[0]
        if annotation_name not in image_files:
            # Si no existe la imagen correspondiente, borrar el archivo XML
            os.remove(os.path.join(annotation_dir, annotation_file))
            print(f'Archivo borrado: {annotation_file}')