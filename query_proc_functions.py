# Now we will create a search function that tokenises and compares the user querry to the dataframe and
# returns top n texts ranked by relevance based on their cosine distance to the dataframe embeddings

from scipy import spatial # calculates vector similarity
import pandas as pd
from openai import OpenAI
import tiktoken
import os
from dotenv import load_dotenv
from df import df

load_dotenv()
OPEN_AI_TOKEN = os.getenv('openai_token')

openai = OpenAI(
  api_key = OPEN_AI_TOKEN,
)

GPT_MODEL = "gpt-3.5-turbo" 
EMBEDDING_MODEL = "text-embedding-ada-002"

# Search function
def strings_ranked_by_relatedness(
    query: str, # custom query
    df: pd.DataFrame, # DataFrame with text and embedding columns (knowledge base)
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), # relatedness function, cosine distance
    top_n: int = 100 # select top n results
) -> tuple[list[str], list[float]]: # Function returns a tuple of two lists, first contains strings, second contains floats
    """Returns strings and relatednesses sorted from largest to smallest"""

    # Send custom query to OpenAI API for tokenization
    query_embedding_response = openai.embeddings.create(
    model=EMBEDDING_MODEL,
    input=query,
    )

    # Received tokenized user query
    query_embedding = query_embedding_response.data[0].embedding

    # Compare user query with each tokenized row of DataFrame
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]

    # Sort the resulting list by descending similarity
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)

    # Transform our list into a tuple of lists
    strings, relatednesses = zip(*strings_and_relatednesses)

    # Return the top n results
    return strings[:top_n], relatednesses[:top_n]

# Now we create an ask function that can accept a user querry, search our database for relevant articles, insert this knowledge
# as a message to chaGPT, send chatGPT a message and recieve a response

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Returns the number of tokens in a string for a given model"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function for generating a request to chatGPT based on a user question and knowledge base
def query_message(
    query: str, # custom query
    df: pd.DataFrame, # DataFrame with text and embedding columns (knowledge base)
    model: str, # model
    token_budget: int # limit on the number of tokens sent to the model
) -> str:
    """Returns a message for GPT with the corresponding source texts extracted from the data frame (knowledge base)."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df) # function for ranking the knowledge base by user query
    # Template instructions for chatGPT
    message = '''Use the below articles about Manchester United F.C to answer the subsequent question. If the answer cannot be found in the articles, give an alternate reply from your knowledge base. If there's stiil no answer, write "I'm sorry, I can only answer questions about Manchester United"'''
    #write "I could not find an answer."'
    # Question Template
    question = f"\n\nQuestion: {query}"

    # Add relevant lines from the knowledge base to the message for chatGPT until we exceed the allowed number of tokens
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str, # custom query
    df: pd.DataFrame = df, # DataFrame with text and embedding columns (knowledge base)
    model: str = GPT_MODEL, # model
    token_budget: int = 4096 - 500, # limit on the number of tokens sent to the model
    print_message: bool = False, # whether to print the message before sending
) -> str:
    """Answers the question using GPT and the knowledge base."""
    # Form a message to chatGPT (function above)
    message = query_message(query, df, model=model, token_budget=token_budget)
    # If the parameter is True, output the message
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Manchester United F.C."},
        {"role": "user", "content": message},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0 # hyperparameter for the degree of randomness when generating text. Affects how the model selects the next word in the sequence.
    )
    response_message = response.choices[0].message.content
    return response_message

