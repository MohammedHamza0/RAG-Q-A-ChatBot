import os
import logging
from pathlib import Path


os.chdir("RAG-Q-A-ChatBot")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")



list_of_files = [
     "DataSet",
     "NoteBook/notebook.ipynb",
     "src/__init__.py",
     "src/helper.py",
     "src/prompt.py",
     "images",
     "app.py",
     ".env",
     ".env.example",
     "store_index.py",
     "requirements.txt"

]

for filepath in list_of_files:
     filepath = Path(filepath)
     filedir, filename = os.path.split(filepath)
     if filedir != "":
          os.makedirs(filedir, exist_ok=True)
          logging.info(f"The file {filename} created")
          
     if (not os.path.exists(filepath)):
          if filename in ["images", "DataSet"]:
               os.makedirs(name=filename, exist_ok=True)
          else:     
               with open(file=filepath, mode="w") as f:
                    pass
                    logging.info(f"The file path: {filepath} created.")
     else:
          logging.info("The file path is already exists.")