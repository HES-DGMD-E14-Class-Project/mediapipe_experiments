import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("ASL_SIGNS_BASE_DIRECTORY")
# Load the train dataframe and label map
df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

# Create a set of unique signs, sorted alphabetically
unique_signs = sorted(df['sign'].unique().tolist())

print(unique_signs)