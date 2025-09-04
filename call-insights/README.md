# Call Insights Agentic System

A robust, scalable agentic system for processing call center audio files and extracting actionable insights using LangGraph and OpenAI.

## 🏗️ Architecture Overview

This system implements a **multi-agent workflow** that processes audio files through several specialized agents:

```
Audio File → Ingestion Agent → Speech Agent → Analysis Agents → Insights
```

### Agent Pipeline:
1. **Ingestion Agent**: Manages file discovery and state transitions
2. **Speech Agent**: Handles transcription and translation
3. **Summarization Agent**: Creates concise summaries
4. **Topic Classification Agent**: Categorizes conversation topics
5. **Key Points Agent**: Extracts main discussion points
6. **Action Items Agent**: Identifies actionable tasks

## 🚀 Key Features

### ✅ **What's Working Well**
- **Proper Agentic Architecture**: Uses LangGraph with StateGraph for workflow orchestration
- **Tool-Based Design**: Well-structured tools with proper error handling
- **State Management**: Type-safe state management with TypedDict
- **Infrastructure Integration**: AWS (S3, Secrets Manager, RDS) integration
- **Modular Design**: Clean separation of concerns

### 🔧 **Improvements Made**
- **Consolidated Codebase**: Single `workflow.py` file instead of duplicate `graph.py`/`graph2.py`
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Monitoring**: Performance tracking and observability
- **Configuration Management**: Centralized configuration system
- **Type Safety**: Better type hints and validation

## 📁 Project Structure

```
call-insights/
├── src/
│   ├── workflow.py              # Main workflow orchestrator
│   ├── Tools/                   # Agent-specific tools
│   │   ├── Ingestion_Agent_Tools.py
│   │   ├── Speech_Agent_Tools.py
│   │   ├── Summarization_Agent_Tools.py
│   │   ├── Topic_Classification_Agent_Tools.py
│   │   ├── Key_Points_Agent_Tools.py
│   │   └── Action_Items_Agent_Tools.py
│   ├── utils/
│   │   ├── prompt_templates.py  # Agent prompts
│   │   └── s3_utils.py         # S3 utilities
│   └── agents/                  # Additional agent modules
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## 🛠️ Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Environment Variables**:
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=ap-southeast-1
```

3. **AWS Configuration**:
- S3 bucket: `experiment2407`
- Secrets Manager: `rds/sachin`
- RDS PostgreSQL database

## 🎯 Usage

### Basic Usage
```python
from src.workflow import CallInsightsWorkflow, WorkflowConfig, WorkflowMonitor

# Initialize with configuration and monitoring
config = WorkflowConfig()
monitor = WorkflowMonitor()
workflow = CallInsightsWorkflow(config=config, monitor=monitor)

# Run the complete workflow
result = workflow.run_workflow()
print("Workflow completed:", result)
```

### Advanced Configuration
```python
# Customize model settings
config.update_model_config(
    model='gpt-4-turbo',
    temperature=0.1,
    max_tokens=8000
)

# Adjust retry settings
config.update_retry_config(
    max_retries=5,
    base_delay=2
)

# Update AWS settings
config.update_aws_config(
    bucket_name='my-custom-bucket',
    region='us-east-1'
)
```

## 🔍 Monitoring & Observability

The system includes comprehensive monitoring:

```python
# Get performance metrics
performance = monitor.get_performance_summary()
print(f"Success Rate: {performance['success_rate']:.2%}")
print(f"Total Executions: {performance['total_executions']}")
print(f"Average Agent Times: {performance['average_agent_times']}")
```

## 🧪 Testing

### Unit Tests
```bash
# Run tests
python -m pytest tests/ -v
```

### Integration Tests
```bash
# Test with sample audio file
python tests/test_integration.py
```

## 🔧 Error Handling

The system implements robust error handling:

- **Retry Logic**: Exponential backoff for transient failures
- **Graceful Degradation**: Continues processing even if some agents fail
- **Error Recovery**: Automatic rollback of file states on failures
- **Comprehensive Logging**: Detailed logs for debugging

## 📊 Performance Optimization

### Current Optimizations:
- **Single File Processing**: Processes one file at a time for reliability
- **State Management**: Efficient state transitions
- **Tool Caching**: Reuses tool instances
- **Connection Pooling**: Efficient database connections

### Future Optimizations:
- **Parallel Processing**: Process multiple files concurrently
- **Streaming**: Real-time processing capabilities
- **Caching**: Cache transcriptions and analysis results
- **Load Balancing**: Distribute processing across multiple instances

## 🔒 Security Considerations

- **AWS IAM**: Proper permissions for S3, Secrets Manager, RDS
- **API Key Management**: Secure storage of OpenAI API keys
- **Data Encryption**: Encrypted storage and transmission
- **Access Control**: Role-based access to different agents

## 🚀 Deployment

### Local Development
```bash
python src/workflow.py
```

### Production Deployment
```bash
# Using Docker
docker build -t call-insights .
docker run -e OPENAI_API_KEY=your_key call-insights

# Using AWS Lambda
# (Deployment scripts in deployment/ directory)
```

## 📈 Scaling Considerations

### Horizontal Scaling:
- **Multiple Instances**: Run multiple workflow instances
- **Queue Management**: Use SQS for job distribution
- **Database Sharding**: Distribute data across multiple databases

### Vertical Scaling:
- **Resource Optimization**: Adjust memory and CPU allocation
- **Model Selection**: Use appropriate model sizes for tasks
- **Caching Strategy**: Implement Redis for caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review the troubleshooting guide

---

**Note**: This system is designed for production use with proper error handling, monitoring, and scalability considerations. The agentic architecture ensures reliable processing of call center audio files while maintaining data integrity and providing actionable insights.
