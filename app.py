# Main page
import streamlit as st 
from utils.RAGPipeline import RAGPipeline

# page configuration
st.set_page_config(
    page_title="Chat with Web",
    page_icon="🧠",
    layout="wide"
)

# Title for page and sidebar
st.title("AI Website Research Assistant (RAG)")
st.caption("Ask questions from website content using Retrieval Augmented Generation")
st.sidebar.title("Provide URLs")

#set session state variables
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "query" not in st.session_state:
    st.session_state.query = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "disabled" not in st.session_state:
    st.session_state["disabled"] = True

# creating dynamic textboxes to provide URLs based on number selection
urls = []
num_urls = st.sidebar.number_input("Number of URLs", min_value=1, max_value=5, value=2)
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url=="":
        continue
    urls.append(url)

# button to process URL
process_url_clicked = st.sidebar.button("Process URLs")

# If "process URL" button is clicked
if process_url_clicked: 
    
    try:
        # if no urls entered
        if not urls:
            st.warning("Please enter at least one URL to process")
        
        #if urls entered
        else:        
            # displaying status of each stage in building pipeline
            with st.sidebar.status("Building RAG Pipeline...", expanded=True) as status:
                pipeline = RAGPipeline(urls)

                status.write("Loading URLs...")
                pipeline.load_documents()

                status.write("Splitting documents...")
                pipeline.split_documents()
                
                status.write("Creating embeddings...")
                pipeline.create_embeddings()

                status.write("Building vector database...")
                pipeline.build_vectorstore()

                pipeline.build()

                status.update(label="Pipeline ready ✅", state="complete")

            st.session_state.pipeline = pipeline

            # 🔥 Reset question + answer
            st.session_state.query = ""
            st.session_state.answer = ""
            st.session_state.disabled=False            
            
    except ValueError as e:
        st.warning(str(e))
        
    except Exception as e:
        st.error(f"Error Building pipeline : {e}")

# user question
query = st.text_input(
    "Question:",
    key="query",
    disabled=st.session_state.disabled
)

# if question is entered
if query:   
    if st.session_state.query:
        if st.session_state.pipeline:
            #st.session_state.answer = st.session_state.pipeline.query(st.session_state.query)
            st.session_state.answer = st.session_state.pipeline.query(query) 

        # response section
        if st.session_state.answer:
                answer = st.session_state.answer["answer"].replace("FINAL ANSWER:", "").strip()               

                 #Answer Section
                with st.container():
                    st.subheader(" 💡 Answer")
                    st.write(answer)
               
                 # Source Section
                sources = st.session_state.answer["sources"]
                if sources:
                    st.subheader("📚 Sources")
                    
                    for i, source in enumerate(sources):
                        title = source.split("/")[-2].replace("-", " ").title()
                        st.markdown(f"[{i+1}] 🔗 [{title}]({source})")

               
                # Evidence section- Retrival chunks 
                with st.expander("🔎 Retrieval Details"):
                    for i, (doc, score) in enumerate(st.session_state.answer["docs_with_scores"]):
                        # Convert distance to similarity (higher = better)
                        similarity = 1 - score  

                        # Decide color based on thresholds
                        if similarity >= 0.7:
                            color = "🟢 Strong match"
                        elif similarity >= 0.5:
                            color = "🟡 Medium match"
                        else:
                            color = "🔴 Weak match"

                        # Display chunk info
                        st.write(f"Chunk {i+1} | Similarity: {round(similarity,3)} {color}")
                        st.progress(min(similarity, 1.0))  # progress bar visualization

                        # Show snippet + source
                        st.write(doc.page_content[:300])
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        st.write("---")              
    else:
        st.warning("Please process URLs first.")





