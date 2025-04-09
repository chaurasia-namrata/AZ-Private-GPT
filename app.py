from flask import Flask, request, jsonify, render_template, Response, send_file, session, redirect, url_for
from openai import AzureOpenAI
from datetime import datetime
from functools import wraps
import json
import os
import uuid
import requests
import bcrypt

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Store conversations and users in memory (in production, use a database)
conversations = {}
users = {}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Configuration - should be moved to environment variables in production
AZURE_CONFIG = {
    "api_version": "2023-12-01-preview",
    "azure_endpoint": "https://joinal-openai.openai.azure.com",
    "api_key": "CPaU2y68J2xCFFh3wWx9PHmO9ORuSxdU9zMtcBAzLmmJex6vyPGnJQQJ99AKACYeBjFXJ3w3AAABACOGnRFh"
}

# Initialize client
client = AzureOpenAI(**AZURE_CONFIG)

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
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in users and bcrypt.checkpw(password.encode('utf-8'), users[email]['password']):
            session['user_id'] = email
            return redirect(url_for('index'))
        return jsonify({'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in users:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Hash password before storing
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users[email] = {
            'name': name,
            'password': hashed,
            'email': email
        }
        
        session['user_id'] = email
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'})

@app.route('/user/profile', methods=['GET'])
@login_required
def get_user_profile():
    user_id = session.get('user_id')
    if user_id in users:
        user = users[user_id]
        return jsonify({
            'name': user['name'],
            'email': user['email']
        })
    return jsonify({'error': 'User not found'}), 404

@app.route('/user/password', methods=['POST'])
@login_required
def change_password():
    user_id = session.get('user_id')
    if user_id not in users:
        return jsonify({'error': 'User not found'}), 404
    
    current_password = request.json.get('current_password')
    new_password = request.json.get('new_password')
    
    if not bcrypt.checkpw(current_password.encode('utf-8'), users[user_id]['password']):
        return jsonify({'error': 'Current password is incorrect'}), 401
    
    users[user_id]['password'] = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    return jsonify({'message': 'Password updated successfully'})

@app.route('/')
@login_required
def index():
    user_id = session.get('user_id')
    user = users.get(user_id)
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
        # Handle EventSource streaming request
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
            'n': int(request.args.get('n', 1))
        }
        
        if not params['message']:
            return jsonify({"error": "Message is required"}), 400
            
        if not params['conversation_id'] or params['conversation_id'] not in conversations:
            return jsonify({"error": "Invalid conversation ID"}), 400

        conversation = conversations[params['conversation_id']]
        messages = conversation['messages']
        
        # Add user message to history
        user_message = {"role": "user", "content": f"{params['message']}\n\nPlease respond using well-formatted markdown."}
        messages.append({"role": "user", "content": params['message']})
        
        def generate():
            try:
                # Generate title if this is the first message
                if conversation['title'] is None:
                    title_response = client.chat.completions.create(
                        model=params['model'],
                        messages=[{
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

                # Generate response
                stream = client.chat.completions.create(
                    model=params['model'],
                    messages=messages,  # Use conversation history
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
                        yield f"data: {json.dumps({'response': content})}\n\n"
                
                # Add assistant message to history
                messages.append({"role": "assistant", "content": assistant_message})
                yield "data: [DONE]\n\n"

            except Exception as e:
                app.logger.error(f"Error in chat: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
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

if __name__ == '__main__':
    app.run(debug=True)