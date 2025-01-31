import csv
import os

# LangChain imports
# Ensure your VertexAI credentials are configured
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser

# Set up your API keys (ideally via environment variables)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")

# Initialize your LLM
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Create a prompt template that asks about feature existence
prompt_template = """
You are given some context information about a competitor's features and offerings:

Context:
{context}

Question:
Does the competitor have the feature called "{feature_name}"? 
Answer with a single word: "Yes" or "No" (no extra text).
"""

prompt = PromptTemplate(
    input_variables=["context", "feature_name"],
    template=prompt_template
)

# Create a chain that will pass the prompt to the OpenAI LLM
chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())

# Initialize Google Search wrapper
search = GoogleSearchAPIWrapper()

###############################################################################
# CSV Reading and Writing
###############################################################################
filename = "input.csv"
output_filename = "output.csv"
competitor_name = "CyberArk"

with open(filename, mode="r", newline="", encoding="utf-8") as csvfile, \
     open(output_filename, mode="w", newline="", encoding="utf-8") as output_file:

    csv_dict_reader = csv.DictReader(csvfile)
    fieldnames = csv_dict_reader.fieldnames

    if fieldnames is None:
        # Handle the case of an empty or incorrectly formatted CSV
        fieldnames = []

    # Add the competitor name as a new column
    if competitor_name not in fieldnames:
        fieldnames.append(competitor_name)

    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over each row in the CSV
    for row in csv_dict_reader:
        feature_name = row["Features"]  # e.g. "Single Sign-On", "MFA", etc.
        print(f"Processing feature: {feature_name}")

        # 1. Build a Google search query
        query = f"{competitor_name} {feature_name}"
        print(f"    Searching Google for: {query}")

        # 2. Run the search (SerpAPI)
        #    `search.run(query)` returns a text blob summarizing the top results.
        #    Adjust how you parse or chunk these results if you need to handle large texts.
        try:
            search_results = search.run(query)
        except Exception as e:
            print(f"    Error during search: {e}")
            search_results = "No search results found or error occurred."

        # 3. Pass the search results to the LLM chain
        #    We ask the model: "Does the competitor have {feature_name}?"
        try:
            response = chain.run({
                "context": search_results,
                "feature_name": feature_name
            })
            # The prompt instructs the model to output "Yes" or "No"
            cleaned_response = response.strip()
            print(f"    Model response: {cleaned_response}")
        except Exception as e:
            print(f"    Error with LLMChain: {e}")
            cleaned_response = "Error"

        # 4. Assign the response to the competitor column in the CSV row
        row[competitor_name] = cleaned_response
        print(f"{feature_name} : {row[competitor_name]}")
        # 5. Write the updated row to the new CSV
        writer.writerow(row)

print(f"\nAll done! Updated CSV with '{competitor_name}' column is written to {output_filename}.")
