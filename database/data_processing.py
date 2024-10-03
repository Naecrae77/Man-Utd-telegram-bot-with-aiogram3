import mwclient 
from openai import OpenAI
import pandas as pd # We will store the knowledge base and the result of tokenization of the knowledge base in the DataFrame
import os
import getpass
from dotenv import load_dotenv
from data_processing_functions import WIKI_SITE, CATEGORY_TITLE, titles_from_category, all_subsections_from_title, clean_section, split_strings_from_subsection, keep_section

# Initialize the MediaWiki object
# WIKI_SITE refers to the English-language part of Wikipedia
site = mwclient.Site(WIKI_SITE)

# Load a section of a given category
category_page = site.pages[CATEGORY_TITLE]
# Get the set of all category titles with one level of nesting
titles = titles_from_category(category_page, max_depth=1)

# Splitting articles into sections
# you'll have to wait a bit, as it takes a while to parse almost 2k articles
wikipedia_sections = []
for title in titles:
    wikipedia_sections.extend(all_subsections_from_title(title))
print(f"{len(wikipedia_sections)} sections have been found on {len(titles)} pages")

# Apply the clean function to all sections using a list generator
wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]

original_num_sections = len(wikipedia_sections)
wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]

# Split sections into parts
MAX_TOKENS = 1600
wikipedia_strings = []
for section in wikipedia_sections:
     wikipedia_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

# Now that we have divided our knowledge base into shorter, self-contained lines, we can calculate embeddings for each line.

EMBEDDING_MODEL = "text-embedding-ada-002" # Tokenization model from OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('openai_token')

client = OpenAI(api_key = OPENAI_API_KEY)

# Function for sending chatGPT string for its tokenization (calculating embeddings)
def get_embedding(text, model="text-embedding-ada-002"):

    return client.embeddings.create(input = [text], model=model).data[0].embedding

df = pd.DataFrame({"text": wikipedia_strings})

df['embedding'] = df.text.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

SAVE_PATH = "./Man United.csv"
# Save the result
df.to_csv(SAVE_PATH, index=False)
