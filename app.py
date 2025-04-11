from flask import Flask, request, jsonify, render_template, Response, send_file, session, redirect, url_for
from database import create_user, verify_user, update_password, get_user_by_email
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import requests
from openai import AzureOpenAI
from datetime import datetime
from functools import wraps
import json
import os
import uuid
import bcrypt

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Store conversations and users in memory (in production, use a database)
conversations = {}
users = {}

# Store extracted PDF text
pdf_contexts = {}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Configuration - should be moved to environment variables in production
AZURE_CONFIG = {
    "api_version": "2023-12-01-preview",
    "azure_endpoint": "https://joinal-openai.openai.azure.com",
    "api_key": "CPaU2y68J2xCFFh3wWx9PHmO9ORuSxdU9zMtcBAzLmmJex6vyPGnJQQJ99AKACYeBjFXJ3w3AAABACOGnRFh"
}

# Azure Document Intelligence configuration
DOCUMENT_INTELLIGENCE_ENDPOINT = "https://doc-intel-joinal.cognitiveservices.azure.com/"
DOCUMENT_INTELLIGENCE_KEY = "E8buNGR5N7u4z1hYX0GduDumQExHqOCHK3MdGN8EsVjgKQHpbOPMJQQJ99ALACYeBjFXJ3w3AAALACOG92Bc"

# Bing Search configuration
BING_SEARCH_KEY = "83eb5b0e0e4142d4a26b500f908b84a9"
BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

def rewrite_image_prompt(query):
    """Rewrite a user query to be more suitable for image generation"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at writing image generation prompts. Your task is to rewrite user queries into detailed, descriptive prompts that will produce better image results. Focus on visual details, style, mood, and technical aspects. Keep the response ONLY to the rewritten prompt, no explanations.\n\nExample:\nUser: 'cat in garden'\nYou: 'A charming tabby cat lounging in a sun-dappled English garden, surrounded by blooming roses and lavender, soft bokeh effect, golden hour lighting, 4K detailed photography'"},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        app.logger.error(f'Error rewriting image prompt: {str(e)}')
        return query

def perform_bing_search(query, count=3):
    if not BING_SEARCH_KEY or not BING_SEARCH_ENDPOINT:
        return []
    
    try:
        # Set up the request
        headers = {
            'Ocp-Apim-Subscription-Key': BING_SEARCH_KEY
        }
        
        params = {
            'q': query,
            'count': count,
            'mkt': 'en-US'
        }
        
        # Make the API call
        response = requests.get(
            BING_SEARCH_ENDPOINT,
            headers=headers,
            params=params
        )
        response.raise_for_status()
        
        # Parse the results
        search_data = response.json()
        print(search_data)
        results = []
        
        # Add web pages
        if 'webPages' in search_data and 'value' in search_data['webPages']:
            for page in search_data['webPages']['value'][:count]:
                results.append({
                    'title': page['name'],
                    'snippet': page['snippet'],
                    'url': page['url']
                })
        
        return results
    except Exception as e:
        app.logger.error(f'Bing search error: {str(e)}')
        return []


# Initialize clientss
client = AzureOpenAI(**AZURE_CONFIG)
document_client = DocumentAnalysisClient(
    endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
)

# Create images directory if it doesn't exist
image_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

# Available models
MODELS = [
    {"id": "gpt-4o", "name": "GPT-4o"},
    {"id": "gpt-4o-mini", "name": "GPT-4o-Mini"}
]

# Middleware to track request timing
@app.before_request
def start_timer():
    request.start_time = datetime.now()

@app.after_request
def add_headers(response):
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Add server timing if available
    if hasattr(request, 'start_time'):
        duration = (datetime.now() - request.start_time).total_seconds() * 1000
        response.headers['Server-Timing'] = f'total;dur={duration:.2f}'
    
    return response

def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return wrapper

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'GET':
        return render_template('login.html')
    
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        if request.is_json:
            return jsonify({'success': False, 'error': 'Email and password are required'})
        return redirect(url_for('login_page'))

    user = verify_user(email, password)
    if user:
        session['user'] = email
        if request.is_json:
            return jsonify({'success': True})
        return redirect(url_for('index'))

    if request.is_json:
        return jsonify({'success': False, 'error': 'Invalid credentials'})
    return redirect(url_for('login_page'))

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'GET':
        return render_template('register.html')
    
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        if request.is_json:
            return jsonify({'success': False, 'error': 'Email and password are required'})
        return redirect(url_for('register_page'))

    if create_user(email, password):
        session['user'] = email
        if request.is_json:
            return jsonify({'success': True})
        return redirect(url_for('index'))

    if request.is_json:
        return jsonify({'success': False, 'error': 'Username already exists'})
    return redirect(url_for('register_page'))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/user/profile', methods=['GET'])
@login_required
def get_user_profile():
    email = session.get('user')
    if email:
        user_data = get_user_by_email(email)
        if user_data:
            return jsonify({
                'username': user_data['email'],
                'email': user_data['email']
            })
    return jsonify({'error': 'User not found'})

@app.route('/user/change-password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    email = session.get('user')

    if verify_user(email, current_password):
        update_password(email, new_password)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid current password'})

@app.route('/')
@login_required
def index():
    email = session.get('user')
    if not email:
        return redirect(url_for('login'))
    user = get_user_by_email(email)
    if not user:
        return redirect(url_for('login'))
    return render_template('index.html', models=MODELS, user=user)

@app.route('/conversations', methods=['GET', 'POST'])
@handle_errors
def manage_conversations():
    if request.method == 'POST':
        # Create new conversation
        conv_id = str(uuid.uuid4())
        conversations[conv_id] = {
            'id': conv_id,
            'title': None,  # Will be generated after first message
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
        return jsonify(conversations[conv_id])
    else:
        # Get all conversations sorted by creation time
        sorted_convs = sorted(conversations.values(), 
                             key=lambda x: x['created_at'], 
                             reverse=True)
        return jsonify(sorted_convs)

@app.route('/conversations/<conv_id>', methods=['GET', 'DELETE'])
@handle_errors
def conversation(conv_id):
    if request.method == 'DELETE':
        if conv_id in conversations:
            del conversations[conv_id]
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Conversation not found'}), 404
    else:
        if conv_id in conversations:
            return jsonify(conversations[conv_id])
        return jsonify({'error': 'Conversation not found'}), 404

@app.route('/generate-image', methods=['POST'])
@handle_errors
def generate_image():
    data = request.json
    prompt = data.get('prompt')
    conversation_id = data.get('conversation_id')
    
    if not prompt or not conversation_id:
        return jsonify({'error': 'Missing prompt or conversation_id'}), 400
    
    if conversation_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Generate image using DALL-E
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1
    )
    
    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]
    
    # Download and save the image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_filename = f'generated_image_{timestamp}.png'
    image_path = os.path.join(image_dir, image_filename)
    
    generated_image = requests.get(image_url).content
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)
    
    # Save to conversation history
    local_image_url = f'/static/images/{image_filename}'
    
    # Create or update conversation
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            'id': conversation_id,
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
    
    # Always update the title for image generation
    conversations[conversation_id]['title'] = f'Image: {prompt[:30]}...' if len(prompt) > 30 else f'Image: {prompt}'
    
    # Add messages
    conversations[conversation_id]['messages'].append({
        'role': 'user',
        'content': f'/imagine {prompt}'
    })
    conversations[conversation_id]['messages'].append({
        'role': 'assistant',
        'content': local_image_url,
        'is_image': True
    })
    
    return jsonify({
        'success': True,
        'image_url': local_image_url
    })

@app.route('/chat', methods=['GET', 'POST'])
@handle_errors
def chat():
    if request.method == 'GET':
        # Get all parameters and session data up front while in request context
        params = {
            'message': request.args.get('message'),
            'conversation_id': request.args.get('conversation_id'),
            'model': request.args.get('model', MODELS[0]['id']),
            'temperature': float(request.args.get('temperature', 0.7)),
            'max_tokens': int(request.args.get('max_tokens', 800)),
            'top_p': float(request.args.get('top_p', 1)),
            'frequency_penalty': float(request.args.get('frequency_penalty', 0)),
            'presence_penalty': float(request.args.get('presence_penalty', 0)),
            'stop': request.args.get('stop'),
            'n': int(request.args.get('n', 1)),
            'enable_search': request.args.get('enable_search') == 'true'
        }
        
        # Get session data
        pdf_context = session.get('pdf_context')
        pdf_filename = session.get('pdf_filename')
        
        def generate():
            
            if not params['message']:
                yield f"data: {json.dumps({'error': 'Message is required'})}\n\n"
                return
                
            if not params['conversation_id'] or params['conversation_id'] not in conversations:
                yield f"data: {json.dumps({'error': 'Invalid conversation ID'})}\n\n"
                return

            conversation = conversations[params['conversation_id']]
            messages = conversation['messages']
            
            # Use PDF context if available
            
            # Prepare messages list
            chat_messages = []
            
            # Add PDF context if available
            if pdf_context:
                chat_messages.append({"role": "system", "content": f"Here is the context from the uploaded PDF: '{pdf_filename}'\n{pdf_context}\n\nPlease use this context to help answer the user's questions when relevant."})
            
            # Add web search results if enabled
            if params.get('enable_search'):
                try:
                    search_results = perform_bing_search(params['message'])
                    if search_results:
                        # Send search results to frontend
                        yield f"data: {json.dumps({'type': 'search_results', 'results': search_results})}\n\n"
                        
                        # Add search context to chat
                        search_context = "I found the following web search results that might be relevant to the user's query. Please use this information to help provide a more accurate and up-to-date response:\n\n"
                        for result in search_results:
                            search_context += f"- {result['title']}\n{result['snippet']}\nSource: {result['url']}\n\n"
                        chat_messages.append({"role": "system", "content": search_context})
                except Exception as e:
                    app.logger.error(f"Error in web search: {str(e)}")
                    # Continue without search results
                    pass
            
            # Check if this is an image generation request
            is_image_request = any(keyword in params['message'].lower() for keyword in ['imagine', 'generate image', 'create image', 'draw', 'picture of'])
            
            if is_image_request:
                # Rewrite the prompt for better image generation
                rewritten_prompt = rewrite_image_prompt(params['message'])
                if rewritten_prompt != params['message']:
                    yield f"data: {json.dumps({'type': 'image_prompt', 'original': params['message'], 'rewritten': rewritten_prompt})}\n\n"
                    # Update the message with the rewritten prompt
                    params['message'] = rewritten_prompt
            
            # Add conversation history
            chat_messages.extend(messages)
            
            # Add user message to history
            user_message = {"role": "user", "content": f"{params['message']}\n\nPlease respond using well-formatted markdown."}
            messages.append({"role": "user", "content": params['message']})
            chat_messages.append(user_message)
            
            # Generate response using OpenAI
            try:
                # Generate title if this is the first message
                if conversation['title'] is None:
                    title_response = client.chat.completions.create(
                        model=params['model'],
                        messages=chat_messages[:1] + [{
                            "role": "system", 
                            "content": "Generate a very brief 2-4 word title for this conversation based on the user's message. Response should be just the title, nothing else."
                        }, {
                            "role": "user", 
                            "content": params['message']
                        }],
                        temperature=0.7,
                        max_tokens=10
                    )
                    conversation['title'] = title_response.choices[0].message.content.strip()
                    yield f"data: {json.dumps({'title': conversation['title']})}\n\n"

                # Perform Bing search if enabled
                search_results = []
                if params.get('enable_search') and 'search' not in messages[0]['role']:
                    search_results = perform_bing_search(user_message)
                    if search_results:
                        search_context = 'Here are some relevant search results:\n'
                        for result in search_results:
                            search_context += f"- {result['title']}\n  {result['snippet']}\n  Source: {result['url']}\n\n"
                        messages.insert(1, {
                            'role': 'system',
                            'content': search_context
                        })

                # Generate response
                stream = client.chat.completions.create(
                    model=params['model'],
                    messages=chat_messages,  # Use conversation history with PDF context
                    temperature=params['temperature'],
                    max_tokens=params['max_tokens'],
                    top_p=params['top_p'],
                    frequency_penalty=params['frequency_penalty'],
                    presence_penalty=params['presence_penalty'],
                    stop=params['stop'],
                    n=params['n'],
                    stream=True
                )

                assistant_message = ""
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        assistant_message += content
                        yield f"data: {json.dumps({'type': 'response', 'content': content})}\n\n"
                
                # Add assistant message to history
                messages.append({"role": "assistant", "content": assistant_message})

                try:
                    # Generate follow-up questions
                    followup_response = client.chat.completions.create(
                        model=params['model'],
                        messages=[{
                            'role': 'system',
                            'content': 'Based on the previous response, generate exactly 3 short, relevant follow-up questions. Format them as a markdown numbered list.'
                        }, {
                            'role': 'user',
                            'content': assistant_message
                        }],
                        temperature=0.7,
                        max_tokens=200,  # Limit tokens for follow-up questions
                        stream=False
                    )

                    followup_questions = followup_response.choices[0].message.content.strip()
                    # Extract questions and clean them up
                    questions = []
                    for q in followup_questions.split('\n'):
                        q = q.strip()
                        if q and q[0].isdigit():  # Check if it starts with a number
                            # Remove the number and any following characters until the actual question
                            q = q[q.find(' ')+1:].lstrip('.-) ')
                            questions.append(q)
                    if questions:  # Only send if we have questions
                        yield f"data: {json.dumps({'type': 'response', 'content': '\n\n'})}\n\n"  # Add spacing
                        yield f"data: {json.dumps({'type': 'followup_questions', 'questions': questions})}\n\n"
                except Exception as e:
                    app.logger.error(f"Error generating follow-up questions: {str(e)}")
                    # Skip follow-up questions on error
                    pass
                finally:
                    yield "data: [DONE]\n\n"

            except Exception as e:
                app.logger.error(f"Error in chat: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    
    else:  # POST request
        data = request.json
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        user_message = {"role": "user", "content": f"{data['message']}\n\nPlease respond using well-formatted markdown."}
        
        response = client.chat.completions.create(
            model=data.get('model', MODELS[0]['id']),
            messages=[user_message],
            temperature=float(data.get('temperature', 0.7)),
            max_tokens=int(data.get('max_tokens', 800)),
            top_p=float(data.get('top_p', 1)),
            frequency_penalty=float(data.get('frequency_penalty', 0)),
            presence_penalty=float(data.get('presence_penalty', 0)),
            stop=data.get('stop'),
            n=int(data.get('n', 1)),
            stream=False
        )
        
        return jsonify({
            'response': response.choices[0].message.content,
            'model': data.get('model'),
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        })

@app.route('/clear-pdf', methods=['POST'])
@login_required
def clear_pdf():
    if 'pdf_context' in session:
        del session['pdf_context']
    return jsonify({'message': 'PDF context cleared'})

@app.route('/upload-pdf', methods=['POST'])
@login_required
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read the PDF file content
    pdf_content = pdf_file.read()
    
    try:
        # Start the analysis
        poller = document_client.begin_analyze_document("prebuilt-document", pdf_content)
        result = poller.result()

        # Extract text from the document
        extracted_text = ''
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + '\n'

        # Store the extracted text and filename in the session
        session['pdf_context'] = extracted_text
        session['pdf_filename'] = pdf_file.filename
        
        return jsonify({
            'message': 'PDF processed successfully',
            'text_length': len(extracted_text),
            'filename': pdf_file.filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)