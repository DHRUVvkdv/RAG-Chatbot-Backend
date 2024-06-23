from dataclasses import dataclass
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from rag_app.get_chroma_db import get_chroma_db
# from get_chroma_db import get_chroma_db


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    sources: List[str]


def query_rag(query_text: str) -> QueryResponse:
    db = get_chroma_db()

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
    response = model.invoke(prompt)
    response_text = response.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Response: {response_text}\nSources: {sources}")

    return QueryResponse(
        query_text=query_text, response_text=response_text, sources=sources
    )


if __name__ == "__main__":
    # query_rag("How much does a landing page cost to develop?")
    # query_rag("What is the email address of Dhruv?") # Passed
    # Response:
    # Based on the context provided, the email address of Dhruv is dhruvvarshney.job@gmail.com
    # query_rag("What awards have Dhruv won?") 
    # Response:
#     Based on the context provided, Dhruv Varshney has won the following awards:

# 1. Winner - Capital One: Best Financial Hack at the VTHacks Hackathon for the Stonks Simulator project.

# 2. Winner – Major League Hacking: Auth0 - Best Use of Auth0 at the VTHacks Hackathon for the Stonks Simulator project.

# 3. Winner - Best Accessibility & Empowerment (Prize Track by University of Virginia) at the HooHacks Hackathon for the Bytes news application project.
    # query_rag("What different courses has Dhruv taken?") 
    # Response:
#     Response: Based on the context provided, Dhruv has taken the following relevant courses:

# 1. Cloud Software Development
# 2. Android Mobile Software Development (Kotlin)
# 3. Intro to Computer Organization (C)
# 4. Intro to Software Design (Java)
# 5. Data Structures and Algorithms

# The context mentions that Dhruv is a Computer Science student at Virginia Tech and lists these as "Relevant Courses" that he has taken.
    # query_rag("How old is Dhruv?") 
    query_rag("abracadabra") 