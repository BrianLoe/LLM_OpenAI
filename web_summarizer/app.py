from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

import asyncio
import pprint
import textwrap
import argparse

from llm_extractor import extract
from schemas import *
from scraper import ascrape_playwright

from warnings import filterwarnings
filterwarnings("ignore")

prompt_template = """Write a concise bullet point summary of the following:


{text}


CONCISE SUMMARY IN BULLET POINTS:"""

bullet_point_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# TESTING
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    
    args = parser.parse_args()
    token_limit = 4000

    # News sites mostly have <span> tags to scrape
    cnn_url = "https://www.cnn.com"
    wsj_url = "https://www.wsj.com"
    nyt_url = "https://www.nytimes.com/ca/"
    vdb_url = "https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/"

    amazon_url = "https://www.amazon.ca/s?k=computers&crid=1LUXGQOD2ULFD&sprefix=%2Caps%2C94&ref=nb_sb_ss_recent_1_0_recent"

    async def scrape_with_playwright(url: str, tags, **kwargs):
        html_content = await ascrape_playwright(url, tags)
        
        separator = '\n\n'
        if 'separator' in kwargs:
            separator = kwargs['separator']
            
        text_splitter = CharacterTextSplitter(separator=separator)
        texts = text_splitter.split_text(html_content)
        
        docs = [Document(page_content=t) for t in texts]
        
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=bullet_point_prompt, verbose=True)

        output_summary = chain.run(docs)
        wrapped_text = textwrap.fill(output_summary, width=100)
        wrapped_text = wrapped_text.replace("\n", " ").replace(". -", ".\n-")
        print(wrapped_text)
     

        # print("Extracting content with LLM")

        # html_content_fits_context_window_llm = html_content[:token_limit]

        # extracted_content = extract(**kwargs,
        #                             content=html_content_fits_context_window_llm)

        # pprint.pprint(extracted_content)

    # Scrape and Extract with LLM
    asyncio.run(scrape_with_playwright(
        url=args.url,
        tags=["h1", "h2", "h3", "span", "p"],
        schema_pydantic=SchemaArticleWebsites,
        separator=" . "
    ))