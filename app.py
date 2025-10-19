# Import librairies
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def get_text_pdf(pdf_documents):
    """Extrait le texte brut d'une liste de fichiers PDF Streamlit.

    Input:
        pdf_documents: Liste d'objets uploadés (st.file_uploader).

    Output:
        text : Le texte concaténé de toutes les pages des PDF.
    """

    text = ""
    if not pdf_documents:
        return text
    for pdf in pdf_documents:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_morceaux_text(text):
    """Découpe un texte en morceaux qui se recouvrent.

    Input:
        text: Texte brut.

    Output:
        morceaux : Liste de segments.
    """
    text_splitter = CharacterTextSplitter(separator = '\n',
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    morceaux = text_splitter.split_text(text)
    return morceaux

def get_vectorstore(morceaux_text):
    """Construit FAISS à partir des embeddings.

    Input:
        morceaux_text: Liste de segments de texte.

    Output:
        vectorstore: Un vectorstore FAISS prêt à interroger.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=morceaux_text, embedding=embeddings)
    return vectorstore

def init_conversation(vectorstore):
    """Crée la chaîne RAG moderne + la mémoire de conversation.

    Input:
        vectorstore: L'index FAISS.

    Output:
        ask: fonction qui exécute le RAG et met à jour la mémoire.
        memoire: mémoire LangChain (messages).
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = vectorstore.as_retriever(search_kwargs={'k' : 4})
    
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ('system', 
         "Réécris la question de l'utilisateur en une question autonome, "
         "en t'appuyant uniquement sur l'historique si nécessaire."),
         MessagesPlaceholder('chat_history'),
         ('human', '{input}')
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu es un assistant RAG. Réponds uniquement à partir du contexte fourni. "
         "Si l'information n'est pas dans le contexte, dis que cette information ne fait pas partie de ta base de données."
         "Si ce sont des salutations tu peux répondre, et dire que tu es la pour répondre dans le contexte, mais tout le temps, tu peux juste saluer.\n\n"
         "Contexte:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm = llm,
        retriever = retriever,
        prompt = rephrase_prompt
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm = llm,
        prompt = qa_prompt 
    )

    rag_chain = create_retrieval_chain(
        retriever = history_aware_retriever,
        combine_docs_chain = combine_docs_chain
    )

    if '_history_store' not in st.session_state:
        st.session_state._history_store = {}

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        store = st.session_state._history_store
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    rag_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    def ask(user_input, session_id):
        """Pose une question au RAG en tenant compte de l'historique.

        Input:
            user_input: Question utilisateur.

        Output:
            result: Un dictionnaire avec au moins les clés 'answer' et 'context'.
        """
        result = rag_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return result
    
    return ask

def main():

    load_dotenv()
    #OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    #os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "OPENAI_PROXY"]:
        if var in os.environ:
            del os.environ[var]

    st.set_page_config(page_title = "Mini RAG Chatbot")

    if 'ask' not in st.session_state:
        st.session_state.ask = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Les fichiers")
        pdf_documents = st.file_uploader("Charger les fichiers en PDF", accept_multiple_files=True)
        if st.button("Charger"):
            with st.spinner("Chargement..."):

                # Récupérer le pdf sous forme de texte
                raw_text = get_text_pdf(pdf_documents)

                # Créer des morceaux du texte
                morceaux_text = get_morceaux_text(raw_text)
                if not morceaux_text:
                    st.warning("Aucun texte a été extrait.")
                    return

                # Créer des vecteurs (texte)
                vectorstore = get_vectorstore(morceaux_text)

                # Initialisation de la conversation
                ask = init_conversation(vectorstore)
                st.session_state.ask = ask
                st.success("Le bot est prêt à répondre aux questions.")


    st.header("Mini RAG : Chatbot")

    # Afficher un historique des messages
    for message in st.session_state.messages:
        role = "user" if isinstance(message,HumanMessage) else 'assistant'
        with st.chat_message(role):
            st.markdown(message.content)

    # Zone de texte
    question_user = st.chat_input("Pose ta question...")

    if question_user:

        # Charger et afficher le message de l'utilisateur
        with st.chat_message('user'):
            st.markdown(question_user)
        st.session_state.messages.append(HumanMessage(question_user))


        # Vérifier que les fichiers ont bien été chargé
        if st.session_state.ask is None:
            with st.chat_message('assistant'):
                st.markdown("Merci ! Charge d'abord les PDF")
            st.session_state.messages.append(AIMessage("Merci ! Charge d'abord des PDF dans la barre latérale"))
            return
        
        # Lancer le programme et afficher la réponse
        with st.chat_message('assistant'):
            with st.spinner("Réflexion..."):
                result = st.session_state.ask(question_user, st.session_state.session_id)
                reponse = result.get('answer', '')
                st.markdown(reponse)

        st.session_state.messages.append(AIMessage(reponse))


    elif question_user:
        st.info("Charger d'abord les PDF.")


if __name__ == '__main__' : 
    main()
