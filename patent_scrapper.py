import streamlit as st
import os
import requests
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
import serpapi
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment variables if .env exists (for local testing)
load_dotenv()

def fetch_top_patents(query, num_results=10, after_date=None, before_date=None, date_type='publish'):
    """
    Searches for the most relevant and recent patents using SerpAPI's Google Patents engine.
    Returns a list of patent IDs, their links, filing dates, titles, and inventors.
    Attempts to fetch up to num_results by adjusting page and num parameters.
    """
    params = {
        "engine": "google_patents",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY"),
    }
    
    if after_date:
        params["after"] = f"{date_type}:{after_date}"
    if before_date:
        params["before"] = f"{date_type}:{before_date}"
    
    client = serpapi.Client(api_key=params["api_key"])
    all_patents = []
    page = 1
    while len(all_patents) < num_results:
        params["page"] = page
        params["num"] = min(100, num_results - len(all_patents))  # Fetch up to 100 or remaining needed
        
        result = client.search(params)
        
        if 'error' in result:
            st.write(f"Error searching patents at page {page}: {result['error']}")
            break
        
        organic_results = result.get('organic_results', [])
        st.write(f"Fetched {len(organic_results)} results at page {page}, total so far: {len(all_patents)}")
        
        if not organic_results:
            st.write(f"No more results found after page {page}")
            break
        
        for res in organic_results:
            patent_id = res.get('patent_id')
            if patent_id:
                clean_id_for_link = patent_id.lstrip('patent/')
                link = res.get('link', f"https://patents.google.com/patent/{clean_id_for_link}")
                filing_date = res.get('filing_date', 'N/A')
                title = res.get('title', 'No title available')
                inventors = res.get('inventor', ['Unknown Inventor'])
                if isinstance(inventors, str):
                    inventors = [inventors]
                all_patents.append({
                    'id': patent_id,
                    'link': link,
                    'filing_date': filing_date,
                    'title': title,
                    'inventors': inventors
                })
        
        page += 1  # Increment page
        
        if len(organic_results) < params["num"]:  # If less than requested num, likely end of results
            break
    
    # Truncate to requested number if more were fetched
    patents = all_patents[:num_results]
    st.write(f"Returning {len(patents)} patents for query: {query}")
    return patents

def fetch_patent_text(patent_id):
    """
    Fetches the full text of a patent using SerpAPI.
    Returns combined abstract, description, and claims as a string, or None if link is missing.
    """
    params = {
        "engine": "google_patents_details",
        "patent_id": patent_id,
        "api_key": os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY"),
    }
    
    # Use the modern serpapi.Client
    client = serpapi.Client(api_key=params["api_key"])
    try:
        result = client.search(params)
    except Exception as e:
        st.write(f"Error fetching details for patent {patent_id}: {str(e)}")
        return None
    
    if 'error' in result:
        st.write(f"API error for patent {patent_id}: {result['error']}")
        return None
    
    # Extract abstract, claims, and description_link
    abstract = result.get('abstract', '')
    claims = ' '.join(result.get('claims', []))
    description_link = result.get('description_link')
    
    if not description_link:
        st.write(f"Description link not found for patent {patent_id}, skipping.")
        return None
    
    # Fetch the description HTML
    try:
        response = requests.get(description_link)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        st.write(f"Failed to fetch description for patent {patent_id}: {str(e)}")
        return None
    
    # Extract text from the description HTML
    description = soup.get_text(separator='\n', strip=True)
    
    # Combine all text
    full_text = f"Abstract: {abstract}\n\nDescription: {description}\n\nClaims: {claims}"
    return full_text

def filter_current_practice_sections(full_text):
    """
    Filters the full patent text to extract sections discussing current practice.
    Searches for keywords like 'background', 'prior art', 'current', 'existing', 'conventional'.
    Extracts paragraphs containing these keywords.
    """
    keywords = [
        r'\bbackground\b', r'\bprior art\b', r'\bcurrent practice\b',
        r'\bexisting method\b', r'\bconventional\b', r'\brelated art\b',
        r'\bstate of the art\b'
    ]
    
    # Split text into paragraphs (assuming double newlines separate them)
    paragraphs = re.split(r'\n{2,}', full_text)
    
    filtered_paras = []
    for para in paragraphs:
        para_lower = para.lower()
        if any(re.search(kw, para_lower, re.IGNORECASE) for kw in keywords):
            filtered_paras.append(para)
    
    if not filtered_paras:
        return full_text  # Fallback to full text if no matches
    
    filtered_text = '\n\n'.join(filtered_paras)
    st.write(f"Filtered text length: {len(filtered_text)} characters")
    return filtered_text

def analyze_patent(patent_id, user_prompt):
    """
    Analyzes the patent text based on the user prompt using xAI Grok API.
    Includes pre-filtering for current practice sections and handles large texts by chunking.
    Outputs structured JSON for table display with retry on JSON parse failure.
    """
    full_text = fetch_patent_text(patent_id)
    if full_text is None:
        st.write(f"Skipping analysis for patent {patent_id} due to missing data.")
        return []  # Return empty list to avoid processing errors downstream
    filtered_text = filter_current_practice_sections(full_text)
    
    # Initialize xAI client
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    
    # Updated system prompt to retain details and enforce strict JSON
    system_message = (
        "You are an expert in patent analysis. Use the provided patent text chunk to answer the query accurately and concisely. "
        "In your summary, retain as much as possible any processing steps (e.g., 'heat component to 40 degrees'), "
        "and any quantitative data referring to cost, materials, or energy use of these processing steps. "
        "If the chunk is incomplete, note what might be missing. "
        "Output ONLY valid JSON with no additional text, strictly following this structure: "
        "{'current_practices': [{'description': 'brief description of the practice', "
        "'process_steps': ['step 1 with details', 'step 2 with quant data'], "
        "'limitations': ['limitation 1', 'limitation 2']}, ...]}"
    )
    
    # Token encoder (approximate for Grok using GPT-4 tokenizer)
    encoder = tiktoken.encoding_for_model("gpt-4")
    
    # Chunk the filtered_text into token-limited parts
    def chunk_text(text, max_tokens=100000):
        words = text.split()
        current_chunk = []
        current_tokens = 0
        for word in words:
            word_tokens = len(encoder.encode(word + " "))  # Approximate
            if current_tokens + word_tokens > max_tokens:
                yield ' '.join(current_chunk)
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        if current_chunk:
            yield ' '.join(current_chunk)
    
    chunks = list(chunk_text(filtered_text))
    analyses = []
    
    for idx, chunk in enumerate(chunks, 1):
        chunk_tokens = len(encoder.encode(chunk))
        st.write(f"Processing chunk {idx}/{len(chunks)} with ~{chunk_tokens} tokens for patent {patent_id}")
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Patent Text Chunk {idx}/{len(chunks)}:\n{chunk}\n\nQuery: {user_prompt}"}
        ]
        
        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )
        analyses.append(response.choices[0].message.content)
    
    combined_analysis = "\n\n".join([f"Chunk {i}: {a}" for i, a in enumerate(analyses, 1)])
    
    synthesis_system = (
        "You are an expert synthesizer. Combine the following chunk analyses into a single coherent JSON output. "
        "Avoid duplicates and organize current practices separately. "
        "Retain all processing steps, quantitative data, and limitations. "
        "Output ONLY valid JSON with no additional text, strictly following this structure: "
        "{'current_practices': [{'description': 'brief description', "
        "'process_steps': ['step 1', 'step 2'], "
        "'limitations': ['lim 1', 'lim 2']}, ...]}"
    )
    
    synthesis_messages = [
        {"role": "system", "content": synthesis_system},
        {"role": "user", "content": f"Chunk analyses:\n{combined_analysis}\n\nOriginal Query: {user_prompt}"}
    ]
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        final_response = client.chat.completions.create(
            model="grok-3",
            messages=synthesis_messages,
            max_tokens=2000,
            temperature=0.5,
        )
        
        raw_response = final_response.choices[0].message.content
        st.write(f"Raw synthesis response (attempt {attempt + 1}/{max_retries + 1}): {raw_response}")
        
        try:
            json_output = json.loads(raw_response)
            practices = json_output.get('current_practices', [])
            st.write("JSON parsed successfully.")
            break
        except json.JSONDecodeError as e:
            st.write(f"JSON parsing error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries:
                st.write("Max retries reached. Attempting to clean response...")
                import re
                potential_json = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if potential_json:
                    try:
                        json_output = json.loads(potential_json.group())
                        practices = json_output.get('current_practices', [])
                        st.write("Cleaned response parsed successfully.")
                        break
                    except json.JSONDecodeError:
                        st.write("Failed to clean and parse response. Returning empty list.")
                        practices = []
                else:
                    st.write("No valid JSON detected. Returning empty list.")
                    practices = []
            else:
                st.write("Retrying synthesis...")

    return practices

def generate_html(all_practices, user_query):
    html_content = """
    <html>
    <head>
        <title>Patent Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ text-align: center; }}
            h2 {{ color: #333; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Patent Analysis Summary for Query: {}</h1>
    """.format(user_query)
    
    for patent_id, data in all_practices.items():
        practices = data['practices']
        link = data['link']
        filing_date = data['filing_date']
        title = data['title']
        inventors = ', '.join(data['inventors'])
        html_content += f"<h2>Patent <a href='{link}'>{patent_id}</a> (Title: {title}, Inventor(s): {inventors}, Filing Date: {filing_date})</h2>"
        
        if practices:
            for i, practice in enumerate(practices):
                html_content += f"<h3>Current Practice {i+1}: {practice['description']}</h3>"
                
                # Process Steps Table
                steps_df = pd.DataFrame({'Process Steps': practice['process_steps']})
                html_content += "<h4>Process Steps</h4>" + steps_df.to_html(index=False)
                
                # Limitations Table
                lims_df = pd.DataFrame({'Limitations': practice['limitations']})
                html_content += "<h4>Limitations</h4>" + lims_df.to_html(index=False)
                
                html_content += "<hr>"
        else:
            html_content += "<p>No current practices extracted.</p><hr>"
    
    html_content += "</body></html>"
    return html_content

# Streamlit App
st.title("Patent Analysis App")

# User inputs for API keys (secure; optional override)
serpapi_key = st.text_input("SERPAPI Key (optional)", type="password")
xai_key = st.text_input("xAI API Key (optional)", type="password")

# Set environment variables with fallback to secrets
if serpapi_key:
    os.environ["SERPAPI_API_KEY"] = serpapi_key
else:
    os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY", "")
if xai_key:
    os.environ["XAI_API_KEY"] = xai_key
else:
    os.environ["XAI_API_KEY"] = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY", "")

# Other customizable inputs
user_query = st.text_input("Search Query (e.g., 'solar panel manufacturing process')", value="solar panel manufacturing process")
analysis_prompt = st.text_input("Analysis Prompt", value="summarize what is reported as current practice")
num_patents = st.slider("Number of Patents to Analyze", min_value=1, max_value=20, value=10)

# Date range controls
date_type = st.selectbox("Date Type for Filtering", options=["publish", "filing", "priority"], index=0)
after_date = st.text_input("After Date (YYYYMMDD, optional)", value="")
before_date = st.text_input("Before Date (YYYYMMDD, optional)", value="")

if st.button("Run Analysis"):
    if not os.getenv("SERPAPI_API_KEY") or not os.getenv("XAI_API_KEY"):
        st.error("Please enter API keys or ensure they are set in Secrets.")
    else:
        try:
            with st.spinner("Fetching and analyzing patents..."):
                patents = fetch_top_patents(
                    user_query, 
                    num_results=num_patents, 
                    after_date=after_date if after_date else None, 
                    before_date=before_date if before_date else None, 
                    date_type=date_type
                )
                all_practices = {}
                
                for patent in patents:
                    patent_id = patent['id']
                    practices = analyze_patent(patent_id, analysis_prompt)
                    if practices:  # Only add if practices exist
                        all_practices[patent_id] = {
                            'practices': practices,
                            'link': patent['link'],
                            'filing_date': patent['filing_date'],
                            'title': patent['title'],
                            'inventors': patent['inventors']
                        }
                
                # Display results in app (optional; shows DataFrames)
                for patent_id, data in all_practices.items():
                    practices = data['practices']
                    filing_date = data['filing_date']
                    title = data['title']
                    inventors = ', '.join(data['inventors'])
                    if practices:
                        df = pd.DataFrame(practices)
                        st.subheader(f"Patent {patent_id} (Title: {title}, Inventor(s): {inventors}, Filing Date: {filing_date})")
                        st.dataframe(df)
                
                # Generate and download HTML
                html_content = generate_html(all_practices, user_query)
                st.download_button("Download HTML Report", html_content, file_name="patent_analysis.html", mime="text/html")
                
                st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Error: {e}")