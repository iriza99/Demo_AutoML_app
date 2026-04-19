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
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    file_age = now - entry.stat().st_mtime
                    if file_age > max_age:
                        if entry.is_dir():
                            shutil.rmtree(entry.path)
                            logging.info(f"Dossier supprimé : {entry.path}")
                        else:
                            os.remove(entry.path)
                            logging.info(f"Fichier supprimé : {entry.path}")
                except Exception as e:
                    logging.error(f"Erreur lors de la suppression de {entry.path} : {e}")

if __name__ == "__main__":
    cleanup_old_files()
