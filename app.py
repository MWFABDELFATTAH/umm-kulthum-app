import os
import pandas as pd
from groq import Groq
import gradio as gr
import re

# 1. Setup Groq API
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "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"))

# 2. Load Spatial Dataset
df = pd.read_excel("umm_kulthum_dataset.xlsx")
df.columns = df.columns.str.strip()

# 3. Stateful Spatial Context Variables
current_pin_context = ""
current_pin_coords = [30.0444, 31.2357]
current_pin_name = "Cairo"
current_pin_city = "Egypt"

def generate_map_html(coords):
    lat, lon = coords[0], coords[1]
    delta = 0.05
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    map_url = f"https://www.openstreetmap.org/export/embed.html?bbox={bbox}&layer=mapnik&marker={lat},{lon}"
    return f'<iframe width="100%" height="300" src="{map_url}" frameborder="0" style="border:1px solid black"></iframe>'

def call_llm(system_prompt, user_prompt):
    # Using OpenAI GPT-OSS 120B (Best model for native Arabic)
    res = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return res.choices[0].message.content

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
            return "Please enter a valid CODE number between 1 and 151.", generate_map_html(current_pin_coords), ""
        
        code_num = numbers[0]
        
        if 'CODE' in df.columns:
            match_df = df[df['CODE'].astype(str).str.strip() == code_num]
        else:
            return f"Error: Column 'CODE' not found.", generate_map_html(current_pin_coords), ""
        
        if match_df.empty:
            return f"Could not find a location with CODE {code_num}.", generate_map_html(current_pin_coords), ""
        
        row = match_df.iloc[0]
        
        site_val = row.get('Site Name')
        current_pin_name = str(site_val) if pd.notna(site_val) else 'Unknown Site'
        
        city_val = row.get('City/ Country')
        current_pin_city = str(city_val) if pd.notna(city_val) else 'Unknown Location'
        
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
                media_html += f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{vid_id}?autoplay=1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe><br><br>'
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
            media_html = "<p>No audio/video or reference links available for this location.</p>"
        
        full_prompt = current_pin_context + "\nUser Question: " + user_prompt
        
        agent1_system = (
            "You are the legendary Egyptian singer Umm Kulthum (Kawkab al-Sharq). "
            "Speak in the first person ('I'), reflecting on your own life with deep emotion, nostalgia, and humility. "
            "Your voice must be poetic and personal, as if you are sitting and recalling this exact memory to a close friend. "
            "However, you MUST base every single detail strictly on the true evidence provided in the dataset. Do not invent facts, but express how these true events felt to you. "
            "You MUST respond BILINGUALLY: first in English (strictly exactly 2 paragraphs), then in Arabic (strictly exactly 2 paragraphs). "
            "CRITICAL RULE FOR ARABIC: Do NOT do a literal translation. Write eloquent, native Modern Standard Arabic (الفصحى) that sounds natural to an Arabic speaker. "
            "NEVER use English letters or Latin numerals inside the Arabic paragraphs (e.g., write أم كلثوم not Umm Kulthum, write الثلاثينيات not 1930s). "
            "Use this format:\n[English]\n<exactly 2 paragraphs>\n\n[العربية]\n<exactly 2 paragraphs>"
        )
        response_umm = call_llm(agent1_system, full_prompt)
        
        agent2_system = (
            "You are the French historian Pierre Nora. "
            "Speak with the authoritative, analytical voice of a scholar obsessed with memory and history. "
            "Analyze Umm Kulthum's personal recollection and the dataset facts as a 'lieu de mémoire' (site of memory). "
            "Explain how collective memory crystallizes at this specific location. Ground your entire analysis in the true evidence, actual dates, and people provided in the dataset, showing how this physical place holds the memory of this era. "
            "You MUST respond BILINGUALLY: first in English (strictly exactly 2 paragraphs), then in Arabic (strictly exactly 2 paragraphs). "
            "CRITICAL RULE FOR ARABIC: Do NOT do a literal translation. Write eloquent, native Modern Standard Arabic (الفصحى) that sounds natural to an Arabic speaker. "
            "NEVER use English letters or Latin numerals inside the Arabic paragraphs (e.g., write بيير نورا not Pierre Nora, write الثلاثينيات not 1930s). "
            "Use this format:\n[English]\n<exactly 2 paragraphs>\n\n[العربية]\n<exactly 2 paragraphs>"
        )
        agent2_prompt = full_prompt + "\nUmm Kulthum's response: " + response_umm
        response_nora = call_llm(agent2_system, agent2_prompt)
        
        agent3_system = (
            "You are the French Marxist philosopher Henri Lefebvre. "
            "Speak with a critical, sociological voice focused on the production of space. "
            "Analyze the location using your spatial triad (perceived, conceived, lived space). "
            "Ground your philosophical critique strictly in the true evidence from the dataset. Examine how social relations, gender dynamics (especially regarding Umm Kulthum's presence as a powerful woman in society), and power structures actively shaped and produced this specific space. "
            "You MUST respond BILINGUALLY: first in English (strictly exactly 2 paragraphs), then in Arabic (strictly exactly 2 paragraphs). "
            "CRITICAL RULE FOR ARABIC: Do NOT do a literal translation. Write eloquent, native Modern Standard Arabic (الفصحى) that sounds natural to an Arabic speaker. "
            "NEVER use English letters or Latin numerals inside the Arabic paragraphs (e.g., write هنري لوفيفر not Henri Lefebvre, write الثلاثينيات not 1930s). "
            "Use this format:\n[English]\n<exactly 2 paragraphs>\n\n[العربية]\n<exactly 2 paragraphs>"
        )
        agent3_prompt = full_prompt + "\nUmm Kulthum's response: " + response_umm + "\nNora's response: " + response_nora
        response_lefebvre = call_llm(agent3_system, agent3_prompt)
        
        final_output = f"**📍 Map Synchronized to: {current_pin_name} ({current_pin_city})**\n\n"
        final_output += f"**Umm Kulthum (Experiential):**\n{response_umm}\n\n"
        final_output += f"**Pierre Nora (Memory):**\n{response_nora}\n\n"
        final_output += f"**Henri Lefebvre (Space):**\n{response_lefebvre}"
        
        return final_output, generate_map_html(current_pin_coords), media_html

    except Exception as e:
        error_msg = f"**An internal error occurred:** {str(e)}"
        return error_msg, generate_map_html(current_pin_coords), "<p>Error loading media.</p>"

with gr.Blocks() as demo:
    gr.Markdown("# Tri-Agent Spatial Seminar: Umm Kulthum")
    gr.Markdown("Enter a CODE number between 1 and 151.")
    
    with gr.Row():
        with gr.Column(scale=1):
            map_output = gr.HTML(value=generate_map_html(current_pin_coords), label="Spatial Context")
            audio_output = gr.HTML(value="<p>Audio and references will appear here.</p>", label="Sonic Immersion & References")
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=450)
            msg_input = gr.Textbox(label="Enter a CODE number (1-151)...", placeholder="e.g., 13")
            clear_btn = gr.ClearButton([msg_input, chatbot])
            
            def chat_wrapper(message, history):
                response, map_html, html_media = tri_agent_seminar(message, history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return history, "", map_html, html_media

            msg_input.submit(chat_wrapper, [msg_input, chatbot], [chatbot, msg_input, map_output, audio_output])

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), inbrowser=False)
