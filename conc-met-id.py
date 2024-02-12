import pandas as pd
import yake
import time
from llama_index.query_engine import KnowledgeGraphQueryEngine
import openai
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import Neo4jGraphStore
openai.api_key = key
import logging
import sys
from llama_index.llms import OpenAI
from llama_index.indices.service_context import ServiceContext



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


url = "bolt://localhost:7687"
username ="neo4j"
password = "password"
# define LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

# Define Neo4j
graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database="neo4j",
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)
df = pd.read_csv('balanced_df.csv')

language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 5
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

# Your sentence
list_of_keywords = []
# Extract keywords with YAKE
# DataFrame to store results
results_df = pd.DataFrame(columns=['Original_Sentence', 'Processed_Result'])

# Iterate over each row in your DataFrame
for index, row in df.iterrows():
    sentence = row['Example']

    # Extract keywords with YAKE
    keywords = custom_kw_extractor.extract_keywords(sentence)
    list_of_keywords = [kw[0].upper() for kw in keywords]

    # Query your knowledge graph
    responser = query_engine.query(f"What metaphors contain or are related to at least one of the words {list_of_keywords}?")
    responser2 = query_engine.query(f"Give me examples of the metaphors that contain or are related to at least one of this words {list_of_keywords}. If the query returns nothing, give me 3 random metaphors and 3 examples of these metaphors from the knowledge base. You must return them in the following format \n <<sentence:example>> \n<<answer:conceptual metaphor>>")#array valori sep da ,
    responser = str(responser)
    responser2 = str(responser)

    # Define the prompts
    gpt_assistant_prompt = "You're a helpful assistant."
    gpt_user_prompt = f"The original question is as follows: given the provided sentence, '{sentence}', what is the conceptual metaphor behind it?\nWe have provided some context in order for you to know more about the keywords contained: '{responser}'\nWe have the opportunity to refine the existing answer using both the existing answer context and your own knowledge, update (expand) or repeat the existing answer if it's good. Provide the answer in the format <<conceptual metaphor>>. If the sentence, to you, has no conceptual metaphor, answer in the format <<no metaphor>>.\n"
#few shots rag
    # Define the messages for the conversation
    messages = [
        {"role": "assistant", "content": gpt_assistant_prompt},
        {"role": "user", "content": gpt_user_prompt}
    ]

    # Set parameters for the completion request
    temperature = 0
    max_tokens = 256
    frequency_penalty = 0.0

    # Create an instance of the OpenAI GPT client
    client = openai.ChatCompletion()

    # Get the response from the model
    response = client.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )
    gpt_result = response.choices[0].message
    # Print the response
    print(response.choices[0].message)
    # Save the result
    results_df = results_df.append({'Original_Sentence': sentence, 'Processed_Result': gpt_result}, ignore_index=True)
    time.sleep(10)  # Sleep for 10 seconds, adjust as needed

results_df.to_csv('processed_results.csv', index=False)

