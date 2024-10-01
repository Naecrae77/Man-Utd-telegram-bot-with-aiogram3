import ast
import pandas as pd

embeddings_path = "https://storage.yandexcloud.net/man-united/Man%20United.csv"
df = pd.read_csv(embeddings_path)
# Convert our embeddings from strings to lists
df['embedding'] = df['embedding'].apply(ast.literal_eval)
