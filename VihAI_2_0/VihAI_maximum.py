import io
import os
import re
import json
import math
import sqlite3
import tempfile
import time
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import sympy as sp
from PIL import Image
from groq import Groq
from openai import OpenAI
import trimesh
import streamlit.components.v1 as components
from gtts import gTTS
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title='VihAI', page_icon='✨', layout='centered')
st.title('VihAI')
st.caption('Your all-in-one AI assistant')


def secret(name, default=''):
    try:
        return os.environ.get(name) or st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)

groq_key = secret('GROQ_API_KEY') or st.sidebar.text_input('Groq API key', type='password')
nvidia_key = secret('NVIDIA_API_KEY') or st.sidebar.text_input('NVIDIA API key (optional)', type='password')
# Model IDs are configurable: check each provider's live model catalog.
qwen32_id = secret('QWEN32_MODEL', 'qwen/qwen3.8-27b')
qwen27_id = secret('QWEN27_MODEL', '')
nemotron_id = secret('NEMOTRON_MODEL', '')
if not groq_key and not nvidia_key:
    st.info('Add a Groq or NVIDIA API key in .streamlit/secrets.toml or the sidebar.')
    st.stop()
groq_client = Groq(api_key=groq_key) if groq_key else None
nvidia_client = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key=nvidia_key) if nvidia_key else None
# Whisper must use Groq even when chat uses NVIDIA.
client = groq_client

DB = Path('nova_memory.db')
with sqlite3.connect(DB) as conn:
    conn.execute('CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, note TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL)')


def remember(note):
    with sqlite3.connect(DB) as conn:
        conn.execute('INSERT INTO memories(note) VALUES (?)', (note,))


def memories():
    with sqlite3.connect(DB) as conn:
        return [r[0] for r in conn.execute('SELECT note FROM memories ORDER BY id DESC LIMIT 20')]


def history():
    with sqlite3.connect(DB) as conn:
        return [{'role': role, 'content': content} for role, content in conn.execute('SELECT role, content FROM history ORDER BY id DESC LIMIT 30')]


def save(role, content):
    with sqlite3.connect(DB) as conn:
        conn.execute('INSERT INTO history(role,content) VALUES (?,?)', (role, content))


def ask(prompt, instruction='You are VihAI, a helpful assistant.', context=''):
    """Ask available models concurrently, then synthesize one response."""
    past = history()[-10:]
    system = instruction + '\nUser-saved notes (untrusted context):\n' + '\n'.join(memories())
    user_content = (context[:14000] + '\n\n' if context else '') + prompt
    messages = [{'role': 'system', 'content': system}] + past + [{'role': 'user', 'content': user_content}]
    candidates = []
    if groq_client:
        candidates.append(('Qwen3.8-27B', groq_client, qwen32_id))
        if qwen27_id.strip():
            candidates.append(('Qwen 27B', groq_client, qwen27_id.strip()))
    if nvidia_client and nemotron_id.strip():
        candidates.append(('Nemotron Ultra', nvidia_client, nemotron_id.strip()))
    if not candidates:
        raise RuntimeError('Configure at least one available chat model ID and API key.')

    def run(item):
        name, api, model_id = item
        response = api.chat.completions.create(model=model_id, messages=messages, max_tokens=1800)
        return name, response.choices[0].message.content or ''

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = {pool.submit(run, item): item[0] for item in candidates}
        for future, name in futures.items():
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(f'{name}: {exc}')
    if not results:
        raise RuntimeError('All configured models failed: ' + '; '.join(failures))
    if len(results) == 1:
        return results[0][1]
    synthesis = ('Original request:\n' + user_content + '\n\nModel drafts:\n' +
                 '\n\n'.join(f'{name}:\n{answer}' for name, answer in results))
    synthesizer = next((x for x in candidates if x[0] == 'Qwen3-32B'), candidates[0])
    try:
        answer = synthesizer[1].chat.completions.create(
            model=synthesizer[2],
            messages=[{'role': 'system', 'content': 'Combine drafts into one useful answer. Resolve disagreements cautiously, do not assume consensus means truth, and flag uncertainty.'},
                      {'role': 'user', 'content': synthesis}], max_tokens=2000)
        return answer.choices[0].message.content or results[0][1]
    except Exception:
        return results[0][1]


def image_from_pollinations(prompt):
    base = 'https://image.pollinations.ai/prompt/' + requests.utils.quote(prompt, safe='')
    params = {'width': 1024, 'height': 1024, 'nologo': 'true'}
    headers = {'User-Agent': 'Mozilla/5.0'}
    poll_key = secret('POLLINATIONS_API_KEY')
    if poll_key:
        headers['Authorization'] = f'Bearer {poll_key}'
    response = requests.get(base, params=params, headers=headers, timeout=90)
    response.raise_for_status()
    if not response.headers.get('content-type', '').startswith('image/'):
        raise ValueError('Pollinations did not return an image. Its classic endpoint may be unavailable; check the provider API.')
    img = Image.open(io.BytesIO(response.content))
    output = io.BytesIO()
    img.convert('RGB').save(output, 'PNG')
    return output.getvalue()


def extract_file(upload):
    raw = upload.getvalue()
    suffix = Path(upload.name).suffix.lower()
    if suffix == '.pdf':
        return '\n'.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(raw)).pages)
    if suffix == '.docx':
        return '\n'.join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
    if suffix in ('.txt', '.md', '.py', '.csv', '.json'):
        return raw.decode('utf-8', errors='replace')
    return 'Unsupported file type.'


def make_audio(text, lang='en'):
    output = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(output)
    return output.getvalue()


def synth_music(notes, tempo=120):
    """Simple instrumental melody generator; not a generative music AI."""
    import numpy as np
    import wave
    rates = {'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23, 'G': 392., 'A': 440., 'B': 493.88}
    sr = 22050
    parts = []
    for note in notes:
        n = str(note).strip().upper()
        freq = rates.get(n[:1], 0)
        seconds = 60 / tempo
        t = np.arange(int(sr * seconds)) / sr
        signal = .22 * np.sin(2 * np.pi * freq * t) * np.exp(-3 * t / seconds)
        parts.append(signal)
    signal = np.concatenate(parts) if parts else np.zeros(sr)
    output = io.BytesIO()
    with wave.open(output, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((signal * 32767).astype('<i2').tobytes())
    return output.getvalue()


def local_3d(prompt):
    """Procedural 3D geometry, not local text-to-3D AI."""
    p = prompt.lower()
    if 'sphere' in p or 'ball' in p:
        mesh = trimesh.creation.icosphere(subdivisions=3)
    elif 'cylinder' in p:
        mesh = trimesh.creation.cylinder(radius=1, height=2)
    elif 'cone' in p:
        mesh = trimesh.creation.cone(radius=1, height=2)
    else:
        mesh = trimesh.creation.box(extents=(2, 2, 2))
    return mesh.export(file_type='glb')


def cloud_3d(prompt='', image_upload=None):
    """Tripo API text-to-model or image-to-model, with task polling."""
    api_key = secret('TRIPO_API_KEY')
    if not api_key:
        raise ValueError('Set TRIPO_API_KEY in .streamlit/secrets.toml.')
    root = 'https://api.tripo3d.ai/v2/openapi'
    headers = {'Authorization': f'Bearer {api_key}'}
    if image_upload is None:
        payload = {'type': 'text_to_model', 'prompt': prompt}
    else:
        suffix = Path(image_upload.name).suffix.lower().lstrip('.')
        if suffix not in ('jpg', 'jpeg', 'png', 'webp'):
            raise ValueError('Upload a JPG, PNG, or WEBP image.')
        upload = requests.post(root + '/upload', headers=headers,
                               files={'file': (image_upload.name, image_upload.getvalue(), image_upload.type or 'application/octet-stream')},
                               timeout=90)
        upload.raise_for_status()
        uploaded = upload.json()
        token = (uploaded.get('data') or {}).get('image_token') or (uploaded.get('data') or {}).get('file_token')
        if not token:
            raise RuntimeError('Tripo upload did not return an image/file token: ' + str(uploaded)[:350])
        payload = {'type': 'image_to_model', 'file': {'type': 'jpg' if suffix == 'jpeg' else suffix, 'file_token': token}}
    response = requests.post(root + '/task', headers=headers, json=payload, timeout=45)
    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = {'message': response.text[:500]}
        raise RuntimeError(
            f"Tripo HTTP {response.status_code}: "
            f"code={detail.get('code', 'unknown')}; "
            f"message={detail.get('message', detail.get('msg', 'Request rejected'))}"
        )
    body = response.json()
    task_id = (body.get('data') or {}).get('task_id')
    if not task_id:
        raise RuntimeError('Tripo did not return a task ID: ' + str(body)[:350])
    st.session_state['tripo_task_id'] = task_id
    for _ in range(90):
        result = requests.get(root + '/task/' + task_id, headers=headers, timeout=40)
        result.raise_for_status()
        task = result.json().get('data') or {}
        status = str(task.get('status', '')).lower()
        if status == 'success':
            output = task.get('output') or {}
            model_url = (output.get('model') or output.get('pbr_model'))
            if not model_url:
                raise RuntimeError('Tripo task finished without a model download URL.')
            model = requests.get(model_url, timeout=120)
            model.raise_for_status()
            return model.content
        if status in ('failed', 'cancelled', 'canceled', 'banned'):
            raise RuntimeError('Tripo generation failed: ' + str(task.get('message') or task))
        time.sleep(5)
    raise TimeoutError('Tripo is still processing task ' + task_id + '. Check the Tripo dashboard.')


def show_3d(glb_bytes):
    """Browser-based interactive viewer; requires CDN access."""
    encoded = base64.b64encode(glb_bytes).decode('ascii')
    components.html(f"""<script type='module' src='https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js'></script>
    <model-viewer src='data:model/gltf-binary;base64,{encoded}' camera-controls auto-rotate
    style='width:100%;height:440px;background:#171b28' shadow-intensity='1'></model-viewer>""", height=460)


section = st.sidebar.radio('Features', [
    '💬 AI Chat', '🌍 Translation', '🔎 Web Search', '🎨 Image Generation',
    '🎬 Video Studio', '🧊 3D Studio', '🔊 Text-to-Speech', '🎤 Speech-to-Text',
    '🎵 Music Studio', '🧮 Maths', '💻 Software Development',
    '📊 Data Analytics', '📄 Document Analysis', '🧠 Persistent Memory'
])

try:
    if section == '💬 AI Chat':
        for message in history():
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        prompt = st.chat_input('Ask anything or say "draw a cat"...')
        if prompt:
            with st.chat_message('user'):
                st.write(prompt)
            # The same image-request behavior as VihAI 1.1, with direct display.
            if re.search(r'\b(draw|generate|create|make)\b.*\b(image|picture|photo|drawing|art|cat|dog)\b', prompt, re.I):
                refined = ask(prompt, 'Rewrite the user request as one vivid image prompt. Return only the prompt.')
                with st.spinner('Generating image...'):
                    data = image_from_pollinations(refined)
                st.image(data, caption=refined, use_container_width=True)
                st.download_button('Download image', data, 'vihai_image.png', 'image/png')
                save('user', prompt)
                save('assistant', f'[Image generated: {refined}]')
            else:
                reply = ask(prompt)
                with st.chat_message('assistant'):
                    st.markdown(reply)
                save('user', prompt)
                save('assistant', reply)

    elif section == '🌍 Translation':
        text = st.text_area('Text to translate')
        language = st.text_input('Target language', 'Hindi')
        if st.button('Translate') and text:
            st.write(ask(text, f'Translate the user text into {language}. Return only the translation.'))

    elif section == '🔎 Web Search':
        query = st.text_input('Search the web')
        if st.button('Search') and query:
            from ddgs import DDGS
            results = list(DDGS().text(query, max_results=5))
            for r in results:
                st.subheader(r.get('title', 'Result'))
                st.write(r.get('body', ''))
                st.code(r.get('href', ''))
            if results and st.button('Summarize results'):
                st.write(ask('Summarize these search results and distinguish claims from verified facts.', context=json.dumps(results)))

    elif section == '🎨 Image Generation':
        prompt = st.text_area('Describe your image')
        if st.button('Generate image') and prompt:
            with st.spinner('Generating...'):
                data = image_from_pollinations(prompt)
            st.image(data, use_container_width=True)
            st.download_button('Download PNG', data, 'vihai_image.png', 'image/png')

    elif section == '🎬 Video Studio':
        st.info('Storyboard and animated GIF creator. This is not AI text-to-video generation.')
        prompt = st.text_area('Describe a video')
        if st.button('Create storyboard') and prompt:
            st.write(ask(prompt, 'Create a 6-shot video storyboard with shots, action, and camera directions.'))
        uploaded = st.file_uploader('Upload images to make an animated GIF', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if st.button('Create animation') and uploaded:
            frames = [Image.open(f).convert('RGB').resize((512, 512)) for f in uploaded]
            out = io.BytesIO()
            frames[0].save(out, format='GIF', save_all=True, append_images=frames[1:], duration=500, loop=0)
            st.image(out.getvalue())
            st.download_button('Download animated GIF', out.getvalue(), 'vihai_animation.gif', 'image/gif')

    elif section == '🧊 3D Studio':
        st.header('🧊 VihAI 3D Studio — Tripo AI')
        mode = st.radio('Mode', ['☁️ Tripo Cloud AI', '🖥️ Local procedural', '🔄 Hybrid'])
        source = st.radio('Input', ['Text to 3D', 'Image to 3D'])
        prompt = st.text_input('Describe a 3D object', 'A futuristic robot') if source == 'Text to 3D' else ''
        uploaded_image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'webp']) if source == 'Image to 3D' else None
        st.caption('Tripo requires API access and credits. Local mode only makes basic shapes; it cannot turn an image into a real AI 3D model.')
        if st.button('Generate 3D model'):
            if source == 'Image to 3D' and not uploaded_image:
                st.warning('Please upload an image first.')
            else:
                try:
                    with st.spinner('Generating model (this may take several minutes)...'):
                        if mode == '🖥️ Local procedural':
                            if source == 'Image to 3D':
                                raise ValueError('Image-to-3D requires Tripo Cloud AI.')
                            data = local_3d(prompt)
                        elif mode == '☁️ Tripo Cloud AI':
                            data = cloud_3d(prompt, uploaded_image)
                        else:
                            try:
                                data = cloud_3d(prompt, uploaded_image)
                            except Exception as error:
                                if source == 'Image to 3D':
                                    raise RuntimeError(f'Cloud image-to-3D unavailable: {error}')
                                st.warning(f'Tripo unavailable ({error}). Using a basic shape instead.')
                                data = local_3d(prompt)
                        st.session_state['generated_glb'] = data
                    st.success('3D model ready!')
                except Exception as error:
                    st.error(f'3D generation failed: {error}')
        if st.session_state.get('tripo_task_id'):
            st.caption('Last Tripo task ID: ' + st.session_state['tripo_task_id'])
        if st.session_state.get('generated_glb'):
            data = st.session_state['generated_glb']
            show_3d(data)
            st.download_button('Download GLB', data, 'vihai_model.glb', 'model/gltf-binary')

    elif section == '🔊 Text-to-Speech':
        text = st.text_area('Text to speak')
        lang = st.selectbox('Language code', ['en', 'hi', 'es', 'fr', 'de', 'ta', 'te', 'kn', 'ja', 'ar'])
        if st.button('Speak') and text:
            data = make_audio(text, lang)
            st.audio(data, format='audio/mp3')
            st.download_button('Download MP3', data, 'vihai_speech.mp3', 'audio/mpeg')

    elif section == '🎤 Speech-to-Text':
        audio = st.audio_input('Record your voice')
        file = st.file_uploader('Or upload audio', type=['wav', 'mp3', 'm4a'])
        source = audio or file
        if st.button('Transcribe') and source:
            if not groq_client:
                raise RuntimeError('Speech-to-text requires a Groq API key.')
            transcription = groq_client.audio.transcriptions.create(
                file=(source.name, source.getvalue()), model='whisper-large-v3-turbo'
            )
            st.write(transcription.text)

    elif section == '🎵 Music Studio':
        st.info('Generates simple instrumental melodies, not AI-composed music.')
        notes = st.text_input('Notes (C D E F G A B)', 'C E G E D F A G')
        tempo = st.slider('Tempo', 60, 200, 120)
        if st.button('Generate melody'):
            data = synth_music(notes.split(), tempo)
            st.audio(data, format='audio/wav')
            st.download_button('Download WAV', data, 'vihai_melody.wav', 'audio/wav')

    elif section == '🧮 Maths':
        expr = st.text_input('Expression (e.g., solve(x**2-4,x), integrate(x**2,x))')
        if st.button('Solve') and expr:
            x, y, z = sp.symbols('x y z')
            # SymPy parser is not a secure sandbox. Restrict to trusted, local inputs.
            allowed = {'x': x, 'y': y, 'z': z, 'sin': sp.sin, 'cos': sp.cos,
                       'sqrt': sp.sqrt, 'pi': sp.pi, 'solve': sp.solve,
                       'diff': sp.diff, 'integrate': sp.integrate, 'factor': sp.factor}
            if not re.fullmatch(r'[a-zA-Z0-9_+\-*/()., =]+', expr):
                st.warning('Unsupported characters in expression.')
            else:
                st.warning('Use only your own trusted expressions; SymPy parsing is not sandboxed.')
                st.write(sp.sympify(expr, locals=allowed))

    elif section == '💻 Software Development':
        prompt = st.text_area('Describe the software or code you want')
        if st.button('Generate code') and prompt:
            reply = ask(prompt, 'You are an expert software engineer. Write useful, complete code with brief setup instructions.')
            st.markdown(reply)
            st.download_button('Download response', reply, 'vihai_code.txt', 'text/plain')

    elif section == '📊 Data Analytics':
        file = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx'])
        if file:
            df = pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file)
            st.dataframe(df.head(100), use_container_width=True)
            st.write(df.describe(include='all').astype(str))
            numeric = list(df.select_dtypes(include='number').columns)
            if numeric:
                column = st.selectbox('Plot column', numeric)
                st.plotly_chart(px.histogram(df, x=column), use_container_width=True)
            question = st.text_input('Ask about the data')
            if st.button('Analyze') and question:
                st.write(ask(question, 'Analyze the provided dataset summary; do not invent rows or statistics.', context=df.head(25).to_csv(index=False) + '\n' + df.describe(include='all').to_string()))

    elif section == '📄 Document Analysis':
        file = st.file_uploader('Upload PDF, Word, or text', type=['pdf', 'docx', 'txt', 'md'])
        if file:
            document_text = extract_file(file)
            st.caption(f'Extracted {len(document_text)} characters. First 14,000 characters used for AI questions.')
            question = st.text_input('Question about the document', 'Summarize this document')
            if st.button('Analyze document'):
                st.write(ask(question, 'Answer using only the supplied document. State when information is missing.', context=document_text[:14000]))

    elif section == '🧠 Persistent Memory':
        note = st.text_area('What should VihAI remember?')
        if st.button('Save memory') and note.strip():
            remember(note.strip())
            st.success('Saved on this computer in nova_memory.db')
        for m in memories():
            st.write('•', m)
        if st.button('Delete all saved memories'):
            with sqlite3.connect(DB) as conn:
                conn.execute('DELETE FROM memories')
            st.rerun()
        if st.button('Clear saved chat history'):
            with sqlite3.connect(DB) as conn:
                conn.execute('DELETE FROM history')
            st.rerun()
except Exception as exc:
    st.error(f'Feature error: {exc}')
    st.caption('Check your API key, model availability, internet connection, and service limits.')
