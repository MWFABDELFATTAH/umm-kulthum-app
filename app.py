import os
import pandas as pd
from groq import Groq
import gradio as gr
import re

# 1. Setup Groq Cloud API
# Make sure your GROQ_API_KEY is set in your environment variables before running.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Load Spatial Dataset
try:
    df = pd.read_excel("umm_kulthum_dataset.xlsx")
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Error loading Excel file: {e}")
    df = pd.DataFrame() # Empty dataframe to prevent crash on startup

# 3. Stateful Spatial Context Variables
current_pin_context = ""
current_pin_coords = [30.0444, 31.2357] # Default to Cairo
current_pin_name = "Cairo"
current_pin_city = "Egypt"

def generate_map_html(coords):
    lat, lon = coords[0], coords[1]
    delta = 0.05
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    map_url = f"https://www.openstreetmap.org/export/embed.html?bbox={bbox}&layer=mapnik&marker={lat},{lon}"
    return f'<iframe width="100%" height="300" src="{map_url}" frameborder="0" style="border:1px solid black"></iframe>'

def call_llm(system_prompt, user_prompt):
    """
    Future-proof LLM caller. Tries multiple models in case one is deprecated.
    """
    models_to_try = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it"
    ]
    
    for model_name in models_to_try:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3, # Lower temperature = more realistic, less hallucinated
                max_tokens=2000
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
            
    return "⚠️ System Error: All AI models are currently unavailable. Please try again later."

def extract_youtube_id(url):
    if not isinstance(url, str) or url.strip() == '' or url.strip() == '*':
        return None
    match = re.search(r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None

def extract_all_urls(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    urls = re.findall(r'(https?://[^\s]+)', text)
    return [url.rstrip('.,;)') for url in urls]

def tri_agent_seminar(user_prompt, history):
    try:
        global current_pin_context, current_pin_coords, current_pin_name, current_pin_city
        match_df = pd.DataFrame()
        
        numbers = re.findall(r'\b(\d+)\b', user_prompt)
        if not numbers:
            return "Please enter a valid CODE number between 1 and 151.\nيرجى إدخال رقم كود صالح بين 1 و 151.", generate_map_html(current_pin_coords), ""
        
        code_num = numbers[0]
        
        if 'CODE' in df.columns:
            match_df = df[df['CODE'].astype(str).str.strip() == code_num]
        else:
            return "Error: Column 'CODE' not found in dataset.", generate_map_html(current_pin_coords), ""
        
        if match_df.empty:
            return f"Could not find a location with CODE {code_num}.\nتعذر العثور على موقع برمز {code_num}.", generate_map_html(current_pin_coords), ""
        
        row = match_df.iloc[0]
        
        site_val = row.get('Site Name')
        current_pin_name = str(site_val) if pd.notna(site_val) else 'Unknown Site'
        
        city_val = row.get('City/ Country')
        current_pin_city = str(city_val) if pd.notna(city_val) else 'Unknown Location'
        
        # Strict Context to prevent hallucinations
        current_pin_context = f"""
        STRICT RULE: You must ONLY use the information provided below. Do not invent or hallucinate any facts, dates, people, or songs.
        You MUST incorporate details from EVERY cell of the provided dataset into your response.
        
        Dataset Row Details:
        CODE: {row.get('CODE', 'Unknown')}
        Site Name: {current_pin_name}
        Alternative / Native Name: {row.get('Alternative / Native Name', 'Unknown')}
        Place Category / Classification: {row.get('Place Category / Classification', 'Unknown')}
        City/ Country: {current_pin_city}
        Event Date / Time Range: {row.get('Event Date / Time Range', 'Unknown')}
        Accuracy Level: {row.get('Accuracy Level', 'Unknown')}
        Event Type: {row.get('Event Type', 'Unknown')}
        Associated People: {row.get('Associated People', 'Unknown')}
        Associated Works / Songs: {row.get('Associated Works / Songs', 'Unknown')}
        Location Info: {row.get('Location Info', 'Unknown')}
        Description: {row.get('Description', 'Unknown')}
        Tags / Keywords: {row.get('Tags / Keywords', 'Unknown')}
        """
        
        try:
            coords_str = str(row.get('Co-ordinates', '30.0444, 31.2357'))
            if ',' in coords_str:
                lat = float(coords_str.split(',')[0].strip())
                lon = float(coords_str.split(',')[1].strip())
            else:
                lat, lon = 30.0444, 31.2357
        except:
            lat, lon = 30.0444, 31.2357
            
        current_pin_coords = [lat, lon]
        
        av_links = extract_all_urls(row.get('Audio/Video Links', ''))
        img_refs = extract_all_urls(row.get('Images / Reference', ''))
        
        media_html = ""
        
        for url in av_links:
            vid_id = extract_youtube_id(url)
            if vid_id:
                media_html += f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{vid_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe><br><br>'
                break
                
        if av_links:
            media_html += "<b>🎵 Audio/Video Links:</b><br>"
            for url in av_links:
                media_html += f'<a href="{url}" target="_blank" style="display: block; color: blue; text-decoration: underline;">{url}</a>'
        
        if img_refs:
            media_html += "<br><b>🖼️ Images / References:</b><br>"
            for url in img_refs:
                media_html += f'<a href="{url}" target="_blank" style="display: block; color: blue; text-decoration: underline;">{url}</a>'
                
        if not media_html:
            media_html = "<p>No audio/video or reference links available for this location.<br>لا توجد روابط صوت/فيديو أو مراجع لهذا الموقع.</p>"
        
        full_prompt = current_pin_context + "\nUser Question: " + user_prompt
        
        # Bilingual Format Rule applied to all agents
        bilingual_format = """
        CRITICAL LANGUAGE INSTRUCTIONS:
        You MUST respond in BOTH English and Standard Arabic (الفصحى).
        Format your response EXACTLY like this:
        🇬🇧 English:
        [Your 2 paragraphs in English here]

        🇸🇦 العربية:
        [Your 2 paragraphs in Standard Arabic here. 
        STRICT ARABIC RULE: Use ONLY Modern Standard Arabic (الفصحى). Do NOT use any colloquial dialects (عامية). 
        Grammar and vocabulary must be perfect, realistic, and professional. Do not make up Arabic words.]
        """
        
        agent1_system = (
            "You are Umm Kulthum. Answer the user question based ONLY on the provided map context. "
            "You MUST incorporate details from EVERY cell of the provided dataset (Site Name, Date, People, Songs, Description, etc.) into your response. "
            "Do NOT hallucinate or invent any information. Speak with a clear, personal, and human touch, as if you are remembering your own life, but do not exaggerate. "
            "Your response MUST be exactly 2 paragraphs long for each language. "
            + bilingual_format
        )
        response_umm = call_llm(agent1_system, full_prompt)
        
        agent2_system = (
            "You are the historian Pierre Nora. Analyze the map spot and Umm Kulthum's response as a 'lieu de mémoire' (site of memory). "
            "You MUST reference the specific details provided in the dataset (dates, people, descriptions) to support your analysis. "
            "Explain how memory crystallizes here and why it matters based strictly on the provided data. "
            "Your response MUST be exactly 2 paragraphs long for each language. "
            + bilingual_format
        )
        agent2_prompt = full_prompt + "\nUmm Kulthum's response: " + response_umm
        response_nora = call_llm(agent2_system, agent2_prompt)
        
        agent3_system = (
            "You are the philosopher Henri Lefebvre. Analyze the map spot using the spatial triad (perceived, conceived, lived space). "
            "You MUST reference the specific details provided in the dataset (event type, location info, descriptions) to support your analysis. "
            "Focus on the social production of space, gender, and power based strictly on the provided data. "
            "Your response MUST be exactly 2 paragraphs long for each language. "
            + bilingual_format
        )
        agent3_prompt = full_prompt + "\nUmm Kulthum's response: " + response_umm + "\nNora's response: " + response_nora
        response_lefebvre = call_llm(agent3_system, agent3_prompt)
        
        final_output = f"**📍 Map Synchronized to / تم مزامنة الخريطة مع: {current_pin_name} ({current_pin_city})**\n\n---\n\n"
        final_output += f"**1. Umm Kulthum (Experiential / التجربة الحية):**\n{response_umm}\n\n---\n\n"
        final_output += f"**2. Pierre Nora (Memory / الذاكرة):**\n{response_nora}\n\n---\n\n"
        final_output += f"**3. Henri Lefebvre (Space / الفضاء):**\n{response_lefebvre}"
        
        return final_output, generate_map_html(current_pin_coords), media_html

    except Exception as e:
        error_msg = f"**An internal error occurred / حدث خطأ داخلي:** {str(e)}"
        return error_msg, generate_map_html(current_pin_coords), "<p>Error loading media.</p>"

# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Tri-Agent Spatial Seminar: Umm Kulthum\n# ندوة الفضاء ثلاثية الوكلاء: أم كلثوم")
    gr.Markdown("Enter a CODE number between 1 and 151 to begin the spatial seminar.\nأدخل رقم كود بين 1 و 151 لبدء الندوة المكانية.")
    
    with gr.Row():
        with gr.Column(scale=1):
            map_output = gr.HTML(value=generate_map_html(current_pin_coords), label="Spatial Context / السياق المكاني")
            audio_output = gr.HTML(value="<p>Audio and references will appear here.<br>ستظهر الروابط الصوتية والمراجع هنا.</p>", label="Sonic Immersion & References / الروابط الصوتية والمراجع")
        
        with gr.Column(scale=2):
            # Using type="messages" for modern Gradio compatibility
            chatbot = gr.Chatbot(height=500, type="messages") 
            msg_input = gr.Textbox(
                label="Enter a CODE number (1-151) / أدخل رقم الكود (1-151)...", 
                placeholder="e.g., 13"
            )
            with gr.Row():
                submit_btn = gr.Button("Submit / إرسال")
                clear_btn = gr.ClearButton([msg_input, chatbot], value="Clear / مسح")
            
            def chat_wrapper(message, history):
                response, map_html, html_media = tri_agent_seminar(message, history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return history, "", map_html, html_media

            msg_input.submit(chat_wrapper, [msg_input, chatbot], [chatbot, msg_input, map_output, audio_output])
            submit_btn.click(chat_wrapper, [msg_input, chatbot], [chatbot, msg_input, map_output, audio_output])

# server_name="0.0.0.0" allows other devices on your university network to connect
demo.launch(server_name="0.0.0.0", inbrowser=True)
