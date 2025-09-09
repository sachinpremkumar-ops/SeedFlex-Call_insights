# 🎯 Call Insights - AI-Powered Call Center Analytics

A sophisticated **multi-agent system** built with **LangGraph** that processes call center audio files and extracts actionable insights through AI-powered analysis. This system transforms raw audio conversations into structured data including summaries, sentiment analysis, action items, and topic classification.

## 🏗️ System Architecture

This system implements a **sequential multi-agent workflow** that processes audio files through specialized AI agents:

```
Audio File → Ingestion → Speech   →    Analysis Agents → Storage & Insights
                ↓           ↓                 ↓
            File Mgmt   Transcription  Parallel Analysis
                            ↓                 ↓
                        Translation    Summarization
                                       Topic Classification
                                       Key Points Extraction
                                       Action Items
                                       Sentiment Analysis
```

### 🤖 Agent Pipeline

The system consists of **8 specialized agents** working in sequence:

1. **🎯 Ingestion Agent** - Manages file discovery and state transitions
2. **🎤 Speech Agent** - Handles transcription and translation
3. **📝 Summarization Agent** - Creates concise conversation summaries
4. **🏷️ Topic Classification Agent** - Categorizes conversation topics
5. **🔑 Key Points Agent** - Extracts main discussion points
6. **✅ Action Items Agent** - Identifies actionable tasks
7. **😊 Sentiment Analysis Agent** - Analyzes emotional tone
8. **💾 Storage Agent** - Stores results and manages embeddings

## 🚀 Key Features

### ✅ **Production-Ready Features**
- **Multi-Agent Architecture**: LangGraph-based workflow orchestration
- **Tool-Based Design**: Well-structured tools with comprehensive error handling
- **State Management**: Type-safe state management with TypedDict
- **AWS Integration**: S3, Secrets Manager, and RDS PostgreSQL
- **LangSmith Integration**: Complete observability and tracing
- **Modular Design**: Clean separation of concerns
- **Error Recovery**: Automatic rollback and retry mechanisms
- **Performance Monitoring**: Built-in tracking and metrics

### 🔧 **Advanced Capabilities**
- **Multi-language Support**: Automatic transcription and translation
- **Sentiment Analysis**: Real-time emotional tone detection
- **Topic Classification**: Intelligent conversation categorization
- **Action Item Extraction**: Automated task identification
- **Vector Embeddings**: Semantic search capabilities
- **HubSpot Integration**: CRM data synchronization

## 📁 Project Structure

```
call-insights/
├── 📄 main.py                          # FastAPI application entry point
├── 📄 langgraph.json                   # LangGraph configuration
├── 📄 pyproject.toml                   # Project dependencies and metadata
├── 📄 requirements.txt                 # Python dependencies
├── 📄 uv.lock                          # UV lock file for reproducible builds
├── 📁 src/                             # Source code directory
│   ├── 📄 graph.py                     # Main workflow orchestrator (ACTIVE)
│   ├── 📄 graph2.py                    # Experimental workflow (COMMENTED)
│   ├── 📄 hubspot_test.py              # HubSpot API integration test
│   ├── 📁 Tools/                       # Agent-specific tools
│   │   ├── 📄 Ingestion_Agent_Tools.py      # File management tools
│   │   ├── 📄 Speech_Agent_Tools.py         # Transcription & translation
│   │   ├── 📄 Summarization_Agent_Tools.py  # Text summarization
│   │   ├── 📄 Topic_Classification_Agent_Tools.py # Topic categorization
│   │   ├── 📄 Key_Points_Agent_Tools.py     # Key points extraction
│   │   ├── 📄 Action_Items_Agent_Tools.py   # Action items identification
│   │   ├── 📄 Sentiment_Analysis_Agent.py   # Sentiment analysis
│   │   └── 📄 Storage_Agent_Tools.py        # Data storage & embeddings
│   ├── 📁 utils/                       # Utility modules
│   │   ├── 📄 prompt_templates.py      # AI agent prompts
│   │   ├── 📄 openai_utils.py          # OpenAI API utilities
│   │   ├── 📄 rds_utils.py             # Database connection utilities
│   │   └── 📄 s3_utils.py              # AWS S3 utilities
│   ├── 📁 sql/                         # Database schema and queries
│   │   ├── 📄 tables_sql.py            # Database table definitions
│   ├── 📁 agents/                      # Additional agent modules (empty)
│   └── 📁 mcp/                         # Model Context Protocol (empty)
├── 📁 static/                          # Static web assets
├── 📁 tests/                           # Test files
└── 📁 logs/                            # Application logs
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.12+**
- **AWS Account** with S3, Secrets Manager, and RDS access
- **OpenAI API Key**
<!-- - **HubSpot API Key** (optional, for CRM integration) -->

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://0fmfrelo8dpli812oabsduzy-admin@bitbucket.org/seedflex/call-transcription-insights.git
cd call-insights

# Install using UV (recommended)
uv sync

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=<your-project-name>
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# ======================================================================================

# S3 Configuration
S3_BUCKET_NAME=experiment2407
S3_PROCESSING_PREFIX=processing/
S3_PROCESSED_PREFIX=processed_latest/

```

## 🔍 LangSmith Setup & Configuration

LangSmith provides comprehensive observability for your LangGraph workflows. Here's how to set it up:

### 1. Create LangSmith Account
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up for a free account
3. Create a new project 

### 2. Get API Key
1. Navigate to Settings → API Keys
2. Create a new API key
3. Copy the key to your `.env` file

### 3. Environment Variables
```bash
# Add these to your .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_api_key_here
LANGCHAIN_PROJECT=call-insights
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 4. LangSmith Features You'll Get

#### 🔍 **Trace Visualization**
- Complete workflow execution traces
- Agent-by-agent performance metrics
- Tool call details and responses
- Error tracking and debugging

#### 📊 **Performance Analytics**
- Execution time per agent
- Token usage tracking
- Cost analysis
- Success/failure rates

#### 🐛 **Debugging Tools**
- Step-by-step execution logs
- Input/output inspection
- Error stack traces
- State transitions

#### 📈 **Monitoring Dashboard**
- Real-time workflow monitoring
- Performance trends
- Alert configuration
- Custom metrics

### 5. Viewing Traces
Once configured, you can view your workflow executions at:
- **Dashboard**: https://smith.langchain.com/projects
- **Traces**: https://smith.langchain.com/traces
- **Sessions**: https://smith.langchain.com/sessions

### Running with LangGraph CLI

```bash

# Start the LangGraph server
uv run langgraph dev

---

*Built using LangGraph, OpenAI, and AWS. Designed for production-scale call center analytics.*