import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader # takes raw text and converts it into a format suitable for lanchain
from langchain_text_splitters import CharacterTextSplitter # split text into meaningful chunks (but here just description for individual books
#from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings #Convert the chunks into document embeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma #Chroma is a vectordb


import gradio as gr


books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])



#RECOMMENDATION PART
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n", ) #we dont want overlap as they are separate
#chunk_size = 0, so that it prioritizes splitting on the separator, rather than the chunk size
documents = text_splitter.split_documents(raw_documents)
#now create embeddings and store in db
db_books = Chroma.from_documents(documents, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")) #This is an encoder based model, hence good for embeddings


def retrieve_semantic_recommendation(
        query: str,
        category: str=None,  #filtering categories
        tone: str = None,  #filtering emotions
        initial_top_k: int = 50,
        final_top_k: int = 16, #that we will list on dashboard
        ) -> pd.DataFrame:

        recs = db_books.similarity_search(query, k=initial_top_k)
        books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
        book_recs = books[books["isbn13"].isin(books_list)]


        #Filtering based on category
        if category!="All":
                book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
        else:
                book_recs = book_recs.head(final_top_k)

        if tone == "Happy":
                book_recs.sort_values(by="joy", ascending=False, inplace=True)
        elif tone == "Surprising":
                book_recs.sort_values(by="surprise", ascending=False, inplace=True)
        elif tone == "Angry":
                book_recs.sort_values(by="anger", ascending=False, inplace=True)
        elif tone == "Suspenseful":
                book_recs.sort_values(by="fear", ascending=False, inplace=True)
        elif tone == "Sad":
                book_recs.sort_values(by="sadness", ascending=False, inplace=True)
        return book_recs



def recommend_books(query: str, category: str, tone: str):
        recommendations = retrieve_semantic_recommendation(query, category, tone)
        results = []

        for _,row in recommendations.iterrows():
                description = row["description"]
                truncated_desc_split = description.split()
                truncated_description = " ".join(truncated_desc_split[:30]) + '....'

                authors_split = row["authors"].split(";")
                if len(authors_split) == 2:
                        authors_str = f"{authors_split[0]} and {authors_split[1]}"
                elif len(authors_split) > 2:
                        authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                else:
                        authors_str = row["authors"]

                caption = f"{row['title']} by {authors_str}: {truncated_description}"
                results.append((row["large_thumbnail"], caption))

        return results




categories = ["All"] + sorted(books["simple_categories"].unique())

tones = ["All"] + ["Happy", "Surprising", "Angry","Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic book recommender")

        with gr.Row():
                user_query = gr.Textbox(label = "Please enter what kind of book you want: ", placeholder = "eg: A story about zombies")
                category_dropdown = gr.Dropdown(choices = categories, label = "Select a category: ", value = "All")
                tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone: ", value = "All")
                submit_button = gr.Button("Find recommendation")

        gr.Markdown("## Recommendations")

        output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

        submit_button.click(fn = recommend_books, inputs = [user_query, category_dropdown, tone_dropdown], outputs = output)


if __name__ == "__main__":
        dashboard.launch(share=True)

