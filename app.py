import streamlit as st # type: ignore
import os
import tempfile
import logging
import re
import json
from urllib.parse import urlparse
from google import genai
from google.genai import types
import httpx
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

def is_valid_url(url):
    """Check if the provided string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_pdf_url(url):
    """Check if the URL points to a PDF file"""
    return url.lower().endswith(('.pdf', '.PDF'))

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_pdf_with_gemini(sources, prompt_text, source_type='file', chat_history=None):
    try:
        responses = []
        
        for source in sources:
            try:
                if source_type == 'file':
                    # Handle uploaded files
                    with st.spinner(f"Processing {source.name}..."):
                        # Save uploaded file to a temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(source.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Read the PDF file as binary
                            with open(tmp_file_path, 'rb') as f:
                                pdf_data = f.read()
                            
                            response_text = process_pdf_data(pdf_data, prompt_text, source.name, chat_history)
                            if response_text:
                                responses.append({
                                    'filename': source.name,
                                    'response': response_text
                                })
                                
                        finally:
                            # Clean up the temporary file
                            try:
                                os.unlink(tmp_file_path)
                            except Exception as e:
                                logging.error(f"Error deleting temporary file: {str(e)}")
                
                elif source_type == 'url':
                    # Handle PDF URLs
                    with st.spinner(f"Processing PDF from URL: {source}..."):
                        try:
                            # Download the PDF from URL
                            response = httpx.get(source, follow_redirects=True)
                            response.raise_for_status()
                            
                            if response.status_code == 200:
                                domain = urlparse(source).netloc
                                filename = f"PDF_from_{domain}.pdf"
                                
                                response_text = process_pdf_data(response.content, prompt_text, filename, chat_history)
                                if response_text:
                                    responses.append({
                                        'filename': filename,
                                        'response': response_text
                                    })
                            
                        except Exception as e:
                            logging.error(f"Error processing URL {source}: {str(e)}")
                            st.error(f"Failed to process PDF from URL: {source}")
            
            except Exception as e:
                logging.error(f"Error processing source {source}: {str(e)}")
                continue
        
        return responses
        
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        st.error(f"Failed to process PDF: {str(e)}")
        return None

def process_pdf_data(pdf_data, prompt_text, source_name, chat_history=None):
    """Helper function to process PDF data with Gemini API"""
    try:
        # Format chat history if provided
        history_context = ""
        if chat_history and len(chat_history) > 1:  # Only include if there's previous conversation
            history_context = "\n\nHistorial de la conversación:"
            for i, chat in enumerate(chat_history[:-1], 1):  # Exclude current question
                history_context += f"\n\nPregunta {i}: {chat['question']}"
                history_context += f"\nRespuesta {i}: {chat['answer']}"
            history_context += "\n\n"

        # Create the prompt with the PDF data and user's question
        base_prompt = """Eres un asistente de inteligencia artificial especializado en licitaciones españolas y tus usuarios son empresas españolas que buscan aplicar a licitaciones.
        Los documentos que te proporcionaré son documentos de licitaciones españolas.
        Debes responder teniendo en cuenta este contexto, evitando responder preguntas no relacionadas.
        Es relevante obtener un resumen de la licitación que incluya el precio de la licitación, requisitos técnicos y administrativos, potencialidad, fórmulas de relevancia con propuestas de valores y resumen de posibles problemas o ventajas para la empresa que aplica.
        Es también relevante proporcionar recomendación sobre si aplicar o no."""
        
        prompt = f"{base_prompt}\n\n{history_context}\n\nPregunta: {prompt_text}"
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
                prompt
            ]
        )
        
        return response.text if response.text else None
        
    except Exception as e:
        logging.error(f"Error in Gemini API call: {str(e)}")
        return None
        
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        st.error(f"Failed to process PDF: {str(e)}")
        return None

def get_gemini_response(prompt_text, file_docs=None, url_docs=None, chat_history=None):
    """
    Process PDF documents using Gemini API and return responses
    """
    all_responses = []
    
    # Process uploaded files if any
    if file_docs:
        file_responses = process_pdf_with_gemini(
            file_docs, 
            prompt_text, 
            source_type='file',
            chat_history=chat_history
        )
        if file_responses:
            all_responses.extend(file_responses)
    
    # Process URLs if any
    if url_docs:
        # Filter out empty or invalid URLs
        valid_urls = [url.strip() for url in url_docs if url.strip() and is_valid_url(url.strip())]# and is_pdf_url(url.strip())]
        if valid_urls:
            url_responses = process_pdf_with_gemini(
                valid_urls, 
                prompt_text, 
                source_type='url',
                chat_history=chat_history
            )
            if url_responses:
                all_responses.extend(url_responses)
    
    if not all_responses:
        return "Error: No se proporcionaron documentos. Por favor, sube un archivo PDF o proporciona una URL de PDF."
    
    # Format responses from all sources
    formatted_responses = []
    for resp in all_responses:
        formatted_responses.append(f"**{resp['filename']}**\n{resp['response']}")
    
    return "\n\n---\n\n".join(formatted_responses)

def user_input(user_question, file_docs=None, url_docs=None, chat_history=None):
    try:
        if not user_question.strip():
            raise ValueError("Pregunta vacía")
            
        if not file_docs and not url_docs:
            st.error("Por favor, sube un archivo PDF o proporciona una URL de PDF primero!")
            return None
            
        return get_gemini_response(
            prompt_text=user_question, 
            file_docs=file_docs, 
            url_docs=url_docs,
            chat_history=chat_history
        )
            
    except Exception as e:
        logging.error(f"Error al procesar la entrada: {str(e)}")
        return "Error: No se pudo procesar la solicitud. Por favor, verifica tu conexión a internet e inténtalo de nuevo."


# Diccionario de preguntas predefinidas con sus versiones detalladas
PREDEFINED_QUESTIONS = {
    "": "",  # Opción vacía por defecto
    "Resumen del documento": "Por favor, proporciona un resumen detallado de los puntos clave de este documento, incluyendo los temas principales, hallazgos importantes y conclusiones relevantes. Incluye ejemplos específicos cuando sea posible.",
    "Tema principal": "Analiza cuidadosamente este documento y describe en detalle cuál es su tema principal. Explica por qué es importante este tema y cómo se desarrolla a lo largo del texto. Incluye subtemas o aspectos clave que apoyen el tema principal.",
    "Hallazgos principales": "Identifica y enumera los hallazgos o conclusiones principales presentados en este documento. Para cada hallazgo, proporciona el contexto necesario y explica su relevancia. Si es posible, incluye datos o estadísticas específicas que respalden estos hallazgos.",
    "Argumentos clave": "Analiza y describe en detalle los argumentos principales presentados en este documento. Explica la lógica detrás de cada argumento, las pruebas presentadas y cómo se relacionan entre sí. Evalúa la solidez de estos argumentos si es posible.",
    "Fechas importantes": "Extrae y enumera todas las fechas, plazos o hitos importantes mencionados en este documento. Para cada uno, proporciona el contexto relevante, incluyendo qué evento o acción está programada y cualquier consecuencia o requisito asociado.",
    "Metodología": "Describe en detalle la metodología utilizada en este documento. Explica los métodos de investigación, herramientas, técnicas o enfoques empleados, así como la justificación para su uso. Incluye cualquier limitación o consideración especial mencionada.",
    "Recomendaciones": "Identifica y describe todas las recomendaciones o elementos de acción propuestos en este documento. Para cada uno, explica su propósito, a quién está dirigido y qué resultados se esperan lograr. Incluye cualquier marco de tiempo o recurso mencionado.",
    "Limitaciones": "Detalla las limitaciones o restricciones mencionadas en este documento. Explica cómo estas limitaciones podrían afectar los hallazgos o conclusiones presentadas. Si se mencionan formas de mitigar estas limitaciones, inclúyelas también.",
    "Datos y estadísticas": "Extrae y resume los datos, estadísticas o cifras clave mencionadas en este documento. Para cada dato, proporciona el contexto, la fuente si está disponible y su relevancia para los hallazgos o conclusiones del documento.",
    "Estructura del documento": "Describe la estructura y organización de este documento. Identifica las secciones principales, su propósito y cómo se relacionan entre sí. Explica cómo esta estructura ayuda a presentar la información de manera efectiva.",
    "Resumen de licitación (JSON)": 'Analiza este documento de licitación y devuelve un JSON con la siguiente estructura en formato Markdown:\n\n```json\n{\n  "porcentaje_recomendacion": "X%",\n  "porcentaje_recomendacion_short_explain": "Breve explicación de 1-2 frases sobre el porcentaje de recomendación",\n  "objeto_contrato": "Descripción detallada del objeto del contrato",\n  "presupuesto": "Presupuesto Base de Licitación (sin IVA)",\n  "solvencia_requerida": "Niveles de solvencia técnica y económica requeridos",\n  "habilitaciones_necesarias": "Lista de habilitaciones, certificaciones o requisitos necesarios",\n  "garantias": "Detalles sobre las garantías requeridas (provisionales, definitivas, etc.)",\n  "ecuaciones": "Fórmulas o criterios de valoración si los hay",\n  "otras_condiciones": "Otras condiciones relevantes de la licitación",\n  "recomendacion": "Análisis detallado y recomendación sobre la conveniencia de participar"\n}\n```\n\nPor favor, completa cada campo con la información relevante extraída del documento. Si algún dato no está disponible, indícalo como No especificado. El campo `porcentaje_recomendacion` debe ser un porcentaje (ej: "75%") y `porcentaje_recomendacion_short_explain` debe ser una explicación concisa de 1-2 frases. Todos los valores de este JSON serán strings que siguen el formato markdown.'
}

# Lista de opciones cortas para el menú desplegable
QUESTION_OPTIONS = [""] + list(PREDEFINED_QUESTIONS.keys())[1:]  # Excluye la clave vacía

def show_landing_page():
    """Display the landing page with file upload and URL input."""
    st.set_page_config(page_title="Analiza tus Licitaciones usando IA", page_icon="📚")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main {
            max-width: 800px;
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            padding: 0.5rem;
            border-radius: 20px;
            font-size: 1.1rem;
            font-weight: bold;
        }
        .stFileUploader {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
        }
        .title {
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title and description
    st.markdown('<h1 class="title">Plataforma de Licitaciones</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>
            Gestiona tus licitaciones de forma eficiente y profesional
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("### 1. Suelta las licitaciones")
    uploaded_files = st.file_uploader(
        "Arrastra tus archivos de licitación aquí",
        accept_multiple_files=True,
        type="pdf",
        key="landing_file_uploader"
    )
    
    # URL input
    st.markdown("### 2. O ingresa URLs de PDFs")
    url_input = st.text_area(
        "Pega las URLs de los PDFs (una por línea)",
        height=100,
        placeholder="https://ejemplo.com/documento1.pdf\nhttps://ejemplo.com/documento2.pdf",
        key="landing_url_input"
    )
    
    # Start button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        start_button = st.button("Comenzar Análisis", use_container_width=True, type="primary")
    
    if start_button:
        if not uploaded_files and not url_input.strip():
            st.error("Por favor, sube al menos un archivo o ingresa una URL para continuar.")
        else:
            # Process URLs
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            valid_urls = [url for url in urls if is_valid_url(url)]
            
            # Update session state
            st.session_state.file_docs = uploaded_files if uploaded_files else []
            st.session_state.url_docs = valid_urls
            st.session_state.show_main_app = True
            st.rerun()
    
    # Feature highlights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Características principales")
    
    # Create three columns for the features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; border-radius: 10px; background-color: #f8f9fa; margin: 0.5rem;'>
            <h3>🔍 Análisis automático</h3>
            <p>Extrae información clave automáticamente de tus documentos con IA avanzada</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; border-radius: 10px; background-color: #f8f9fa; margin: 0.5rem;'>
            <h3>⚡ Carga rápida</h3>
            <p>Sube múltiples archivos PDF o enlaces en segundos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; border-radius: 10px; background-color: #f8f9fa; margin: 0.5rem;'>
            <h3>🔗 Enlaces directos</h3>
            <p>Conecta directamente con documentos en la nube mediante URLs</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add some space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

def extract_json_from_markdown(text):
    """Extract JSON content from markdown code blocks."""
    try:
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # If no markdown code block, try to parse the whole text as JSON
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error(f"Error parsing JSON: {str(e)}")
        return None

def display_json_cards(json_data):
    """Display JSON data as cards, with percentage card in the sidebar and others in main view."""
    if not json_data or not isinstance(json_data, dict):
        return st.markdown("No se pudo mostrar la información en formato de tarjetas.")
    
    # Field labels in Spanish
    field_labels = {
        "porcentaje_recomendacion": "Porcentaje de Recomendación",
        "objeto_contrato": "Objeto del Contrato",
        "presupuesto": "Presupuesto",
        "solvencia_requerida": "Solvencia Requerida",
        "habilitaciones_necesarias": "Habilitaciones Necesarias",
        "garantias": "Garantías",
        "ecuaciones": "Fórmulas de Valoración",
        "otras_condiciones": "Otras Condiciones",
        "recomendacion": "Recomendación"
    }
    
    # Field icons (using emojis)
    field_icons = {
        "porcentaje_recomendacion": "📊",
        "objeto_contrato": "📑",
        "presupuesto": "💰",
        "solvencia_requerida": "📈",
        "habilitaciones_necesarias": "📋",
        "garantias": "🔒",
        "ecuaciones": "📝",
        "otras_condiciones": "📌",
        "recomendacion": "💡"
    }
    
    # Display percentage card in the sidebar
    if "porcentaje_recomendacion" in json_data and json_data["porcentaje_recomendacion"].lower() != "no especificado":
        with st.sidebar:
            st.markdown(
                f"""
                <div style='
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 1.25rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    border-left: 4px solid #4a6fa5;
                '>
                    <div style='
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        text-align: center;
                    '>
                        <span style='
                            font-size: 2.5rem; 
                            color: #4a6fa5;
                            margin-bottom: 0.5rem;
                        '>
                            {field_icons.get("porcentaje_recomendacion", '📊')}
                        </span>
                        <div style='
                            color: #2c3e50;
                            font-weight: 600;
                            font-size: 1.2rem;
                            margin-bottom: 0.5rem;
                        '>
                            {field_labels.get("porcentaje_recomendacion", "Porcentaje de Recomendación")}
                        </div>
                        <div style='
                            color: #4a6fa5; 
                            font-size: 2.5rem;
                            font-weight: bold;
                            line-height: 1;
                            margin: 0.5rem 0;
                        '>
                            {json_data["porcentaje_recomendacion"]}
                        </div>
                        {f'<div style="'
                         f'color: #6c757d; '
                         f'font-size: 0.9rem; '
                         f'margin-top: 0.5rem; '
                         f'line-height: 1.4; '
                         f'padding: 0.5rem 1rem; '
                         f'background-color: rgba(74, 111, 165, 0.1); '
                         f'border-radius: 8px; '
                         f'max-width: 100%; '
                         f'word-wrap: break-word;"' 
                         f'>{json_data.get("porcentaje_recomendacion_short_explain", "")}' 
                         f'</div>' 
                         if json_data.get("porcentaje_recomendacion_short_explain") else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Display other cards in the main view
    for key, value in json_data.items():
        if key != "porcentaje_recomendacion" and key in field_labels and value and value.lower() != "no especificado":
            # Create a card for each field
            st.markdown(
                f"""
                <div style='
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 1.25rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    border-left: 4px solid #4a6fa5;
                '>
                    <div style='
                        display: flex;
                        align-items: flex-start;
                        margin-bottom: 0.75rem;
                    '>
                        <span style='
                            font-size: 1.5rem; 
                            margin-right: 0.75rem;
                            color: #4a6fa5;
                        '>
                            {field_icons.get(key, '📄')}
                        </span>
                        <div>
                            <div style='
                                color: #2c3e50;
                                font-weight: 600;
                                font-size: 1.05rem;
                                margin-bottom: 0.25rem;
                            '>
                                {field_labels.get(key, key.replace('_', ' ').title())}
                            </div>
                            <div style='
                                color: #34495e; 
                                line-height: 1.6;
                                font-size: 0.95rem;
                            '>
                                {value}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def show_main_app():
    """Display the main chat application."""
    st.header("Resultado del Análisis")
    
    # Auto-run the JSON summary question if this is the first time opening the chat
    if "auto_run_complete" not in st.session_state:
        st.session_state.auto_run_complete = False
    
    # Store JSON summary data separately for display
    if "json_summary_data" not in st.session_state:
        st.session_state.json_summary_data = None
    
    # Check if we need to auto-run the JSON summary
    docs_available = st.session_state.get("file_docs") or st.session_state.get("url_docs")
    if not st.session_state.auto_run_complete and docs_available:
        st.session_state.auto_run_complete = True
        json_question = "Resumen de licitación (JSON)"
        if json_question in PREDEFINED_QUESTIONS:
            # Add the question to chat history with a placeholder answer
            st.session_state.chat_history.append({
                "question": PREDEFINED_QUESTIONS[json_question],
                "answer": "Procesando resumen de licitación...",
                "is_json_summary": True
            })
            # Process the question and update the answer
            response = user_input(
                user_question=PREDEFINED_QUESTIONS[json_question],
                file_docs=st.session_state.get("file_docs"),
                url_docs=st.session_state.get("url_docs"),
                chat_history=st.session_state.chat_history
            )
            if st.session_state.chat_history and st.session_state.chat_history[-1]["answer"] == "Procesando resumen de licitación...":
                st.session_state.chat_history[-1]["answer"] = response or "No se pudo generar el resumen de la licitación"
                # Extract and store JSON data for display
                if response:
                    st.session_state.json_summary_data = extract_json_from_markdown(response)
            st.rerun()
    
    # Display JSON summary cards at the top if available
    if st.session_state.json_summary_data:
        display_json_cards(st.session_state.json_summary_data)
        st.markdown("---")  # Add a separator line
    
    # Display chat interface
    st.header("Plataforma de Licitaciones")
    
    # Display chat history (excluding JSON summary from chat)
    if st.session_state.chat_history:
        st.subheader("Historial de Conversación")
        for chat in st.session_state.chat_history:
            # Skip displaying the JSON summary in the chat
            if chat.get("is_json_summary"):
                continue
                
            # Check if this is a predefined question
            is_predefined = any(
                chat["question"] == question_text 
                for question_text in PREDEFINED_QUESTIONS.values()
            )
            
            # Show messages in the correct order
            if not is_predefined:
                # For regular questions, show question first, then answer
                with st.chat_message("user"):
                    st.markdown(chat["question"])
                with st.chat_message("assistant"):
                    st.markdown(chat["answer"])
            else:
                # For other predefined questions, only show the answer
                with st.chat_message("assistant"):
                    st.markdown(chat["answer"])
            
            st.divider()

def main():
    try:
        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "file_docs" not in st.session_state:
            st.session_state.file_docs = []
        if "url_docs" not in st.session_state:
            st.session_state.url_docs = []
        if "predefined_question_trigger" not in st.session_state:
            st.session_state.predefined_question_trigger = ""
        if "show_main_app" not in st.session_state:
            st.session_state.show_main_app = False
        
        # Show main app or landing page based on state
        if not st.session_state.show_main_app:
            show_landing_page()
            return
            
        # Main app interface
        show_main_app()
        
        # Sidebar will only contain the percentage card, which is handled by display_json_cards
        # Clear any existing sidebar content
        st.sidebar.empty()
        
        # Add a clean header to the sidebar
        st.sidebar.markdown("""
            <style>
                .sidebar .sidebar-content {
                    background-color: #f8f9fa;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Add a button to go back to the landing page at the bottom of the sidebar
        if st.sidebar.button("Cargar nuevos documentos"):
            st.session_state.show_main_app = False
            st.rerun()



        # Menú desplegable de preguntas predefinidas
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            selected_question_key = st.selectbox(
                "Selecciona una pregunta predefinida:",
                QUESTION_OPTIONS,
                key="predefined_question_select",
                help="Elige una pregunta para hacer sobre tus documentos"
            )
            
            # Botón para hacer la pregunta seleccionada
            if st.button("Hacer Pregunta Seleccionada") and selected_question_key:
                # Usamos la versión detallada de la pregunta
                st.session_state.predefined_question_trigger = PREDEFINED_QUESTIONS[selected_question_key]
        
        # Entrada de chat en la parte inferior
        user_question = st.chat_input("Pregunta sobre tus documentos:")
        
        # Check if a predefined question was triggered and no manual input was provided
        if not user_question and st.session_state.predefined_question_trigger:
            user_question = st.session_state.predefined_question_trigger
            # Clear the trigger to avoid resubmission
            st.session_state.predefined_question_trigger = ""
        
        if user_question and (st.session_state.file_docs or st.session_state.url_docs):
            # Add user question to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": "Procesando..."
            })
            
            # Get response from Gemini with chat history
            response = user_input(
                user_question=user_question, 
                file_docs=st.session_state.file_docs,
                url_docs=st.session_state.url_docs,
                chat_history=st.session_state.chat_history[-10:]  # Only send last 10 messages to avoid context window issues
            )
            
            # Update the last message with the actual response
            if st.session_state.chat_history and st.session_state.chat_history[-1]["answer"] == "Procesando...":
                st.session_state.chat_history[-1]["answer"] = response or "No se generó ninguna respuesta"
            
            # Rerun to update the UI
            st.rerun()

    except Exception as e:
        logging.critical(f"Error crítico en la aplicación: {str(e)}")
        st.error("La aplicación encontró un error crítico. Por favor, recarga la página.")


if __name__ == "__main__":
    main()

