import os
import time
import shutil
import logging

logging.basicConfig(
    filename='cleanup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

FOLDERS_TO_CLEAN = {
    'uploads': 3600,         # 1h
    'models': 86400,         # 24h
    'decompression': 3600,   # 1h
    'prediction': 3600       # 1h
}

def cleanup_old_files():
    now = time.time()
    for folder, max_age in FOLDERS_TO_CLEAN.items():
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                file_age = now - os.path.getmtime(file_path)
                if os.path.isdir(file_path):
                    if file_age > max_age:
                        shutil.rmtree(file_path)
                        logging.info(f"Dossier supprimé : {file_path}")
                else:
                    if file_age > max_age:
                        os.remove(file_path)
                        logging.info(f"Fichier supprimé : {file_path}")
            except Exception as e:
                logging.error(f"Erreur lors de la suppression de {file_path} : {e}")

if __name__ == "__main__":
    cleanup_old_files()
