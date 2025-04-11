# PrivateGPT

A secure, enterprise-grade AI assistant that keeps your organization's data private. Built for companies concerned about data privacy when using AI tools, PrivateGPT provides a controlled environment where sensitive information never leaves your infrastructure.

## Why PrivateGPT?

In today's AI-driven workplace, organizations face a critical challenge: employees often use public AI assistants and copilots for their work, potentially exposing sensitive company data that could:
- Be used to train future public AI models
- Leak intellectual property or confidential information
- Violate data protection regulations and compliance requirements
- Compromise competitive advantages

PrivateGPT solves this by providing:
- A fully controlled, internal AI environment
- No data sharing with external AI providers
- Complete audit trail of AI interactions
- Enterprise-grade security and authentication
- Compliance with data protection regulations

## Features

- 🚀 Real-time streaming responses
- 🎨 Beautiful, responsive UI with dark mode support
- 💬 Conversation management and history
- ⚙️ Customizable model parameters
- 🔒 Secure authentication with email-based login
- 📝 Markdown support with syntax highlighting
- 🎯 Smart conversation titling
- 🔄 Collapsible conversation sidebar
- 🌐 Web search integration
  - Toggle web search with a single click
  - Enhanced responses with real-time web results
  - Seamless Bing Search API integration
- 📎 PDF document upload and context support
- ❓ Smart follow-up questions
  - Auto-generated relevant questions
  - Interactive question bubbles
  - Click-to-ask functionality
- 🎭 Dynamic message status indicators
  - Upload status feedback
  - Processing state indicators

## Tech Stack

- **Backend:**
  - Flask (Python web framework)
  - SSE (Server-Sent Events) for real-time streaming
  - SQLite for data persistence
  - JWT for authentication
  - Azure Bing Search SDK for web search

- **Frontend:**
  - TailwindCSS for styling
  - Marked.js for markdown rendering
  - highlight.js for code syntax highlighting
  - Pure JavaScript (No framework dependencies)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/InternalGPT.git
cd InternalGPT
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
flask db upgrade
```

## Usage

1. Start the development server:
```bash
flask run
```

2. Access the application at `http://localhost:5000`

## Configuration

### Environment Variables

- `SECRET_KEY`: Application secret key
- `DATABASE_URL`: SQLite database URL
- `MODEL_NAME`: Default GPT model to use
- `API_KEY`: Your API key for the model provider
- `AZURE_FORM_RECOGNIZER_ENDPOINT`: Azure Form Recognizer endpoint for PDF processing
- `AZURE_FORM_RECOGNIZER_KEY`: Azure Form Recognizer key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_KEY`: Azure OpenAI key
- `BING_SEARCH_KEY`: Bing Web Search API key for enhanced context

### Available Models

PrivateGPT supports multiple GPT models to balance capability, speed, and cost:

- **GPT-4 Turbo**
  - Most capable model for complex tasks
  - 128K context length
  - Best for: Complex analysis, coding, and creative tasks
  - Cost: $0.01 per 1K tokens

- **GPT-4**
  - High capability with standard context
  - 8K context length
  - Best for: Detailed reasoning and expert tasks
  - Cost: $0.03 per 1K tokens

- **GPT-3.5 Turbo**
  - Fast and efficient for most tasks
  - 16K context length
  - Best for: Day-to-day queries and simpler tasks
  - Cost: $0.002 per 1K tokens

### Model Parameters

Fine-tune model behavior with adjustable parameters:
- Temperature (0-2): Controls randomness
- Top P (0-1): Nucleus sampling threshold
- Frequency Penalty (0-2): Reduces repetition
- Presence Penalty (0-2): Encourages topic diversity
- Max Tokens: Limits response length
- Stop Sequences: Custom completion markers

## Project Structure

```
InternalGPT/
├── app.py              # Main application file
├── templates/          # HTML templates
│   └── index.html     # Main UI template
├── static/            # Static assets
├── utils.py           # Utility functions
├── data/             # Data storage
│   └── conversations/ # Conversation files
└── requirements.txt   # Python dependencies
```

## Features in Detail

### Real-time Streaming

The application uses Server-Sent Events (SSE) to stream AI responses in real-time, providing a smooth chat-like experience.

### Conversation Management

- Create and manage multiple conversations
- Automatic conversation titling
- Collapsible sidebar for better space management
- Delete conversations
- Clear chat history

### UI/UX Features

- Responsive design that works on all device sizes
- Dark mode support
- Beautiful animations and transitions
- Code syntax highlighting
- Markdown rendering with support for:
  - Headers
  - Lists
  - Code blocks
  - Links
  - Emphasis
  - Tables
  - Blockquotes

### Security & Privacy

- **Authentication & Authorization**
  - JWT-based authentication
  - Role-based access control
  - Secure session management

- **Data Protection**
  - All data stays within your infrastructure
  - No external model training
  - Complete data isolation

- **Application Security**
  - Password hashing with bcrypt
  - Input sanitization
  - XSS protection
  - CSRF protection
  - Rate limiting

- **Compliance Features**
  - Audit logging of all interactions
  - Data retention controls
  - Usage monitoring
  - Cost tracking per department

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Flask and TailwindCSS
- Uses Marked.js for markdown rendering
- Uses highlight.js for code syntax highlighting
