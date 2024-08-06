# Development of an AI-assisted Chatbot for a Water Monitoring Environment Lab

## ABSTRACT

The Learning Enhanced Watershed Assessment System (LEWAS) at Virginia Tech has accumulated vast amounts of data and research publications over a decade. To efficiently onboard new students and serve as a comprehensive resource, an AI-assisted chatbot was developed. This paper describes the development, implementation, and evaluation of the LEWAS chatbot, which utilizes advanced natural language processing techniques and cloud computing to provide interactive and efficient access to LEWAS information.

## KEYWORDS

LEWAS, AI-assisted chatbot, environmental monitoring, water quality, knowledge transfer, retrieval-augmented generation, AWS, vector database

## INTRODUCTION

Environmental monitoring labs generate significant amounts of data and research over time, creating challenges in knowledge transfer and information accessibility. The Learning Enhanced Watershed Assessment System (LEWAS) at Virginia Tech is an interdisciplinary environmental real-time water monitoring lab that has faced similar challenges. This paper presents the development of an AI-assisted chatbot designed to address these issues by providing an efficient and interactive means of accessing LEWAS information.

## LEWAS LAB

The LEWAS lab is an interdisciplinary environmental monitoring facility at Virginia Tech. Over the years, it has expanded through contributions from students across various departments, generating extensive water and weather data along with numerous research publications. The lab's interdisciplinary nature and the volume of information it produces necessitate an efficient system for knowledge dissemination and transfer.

## RESEARCH METHOD

### System Architecture

The LEWAS chatbot system consists of two main components:

1. Document Processing and Vector Storage Flow:
   - Ingests documents (PDFs, research papers, lab reports)
   - Converts documents to vector embeddings
   - Stores embeddings in a Pinecone vector database
   - Tracks processed files using Amazon DynamoDB
   - Uses S3 for document storage

2. LEWAS Chatbot System Architecture:
   - Handles user queries through a Streamlit-based frontend
   - Processes queries using natural language processing techniques
   - Utilizes AWS Lambda for serverless computing
   - Employs a FastAPI server for request handling
   - Converts queries to embeddings using AWS Bedrock
   - Performs similarity search in the vector database
   - Generates responses using a large language model (LLM)

### Adding Features

The chatbot implements several key features:
- Retrieval-Augmented Generation (RAG) system for combining retrieved information with generated responses
- User-friendly Streamlit interface
- Integration of multiple AWS services
- Scalable architecture to handle 100+ concurrent users
- Efficient information retrieval using vector search

### Adding Documents

The system is designed to easily integrate new documents:
- Documents are processed and converted to vector embeddings
- Embeddings are stored in the Pinecone vector database
- Processed files are tracked using Amazon DynamoDB
- Documents are stored in Amazon S3 for easy retrieval

### System Side Optimizations

Several optimizations were implemented on the system side:
- Use of AWS Lambda for serverless computing to handle varying loads
- Implementation of a FastAPI server for efficient request handling
- Utilization of AWS Bedrock for LLM and embedding generation

### Data Information

The chatbot has access to a wide range of LEWAS data, including:
- Water quality measurements
- Weather data
- Research publications from undergraduate and graduate students
- Information about LEWAS operations and components

### User Feedback

To improve the chatbot's performance and usability:
- Implement a feedback mechanism within the Streamlit interface
- Regularly collect and analyze user feedback
- Use feedback to refine the chatbot's responses and functionality

### Testing

A comprehensive testing strategy was developed:
- Unit tests for individual components
- Integration tests for the entire system
- Load testing to ensure the system can handle 100+ concurrent users
- Regular evaluation of response accuracy and relevance

### Optimizing Parameters

Several parameters were optimized to improve performance:
- Chunk size for document processing
- Number of chunks (topk) retrieved for each query
- Model selection for embedding generation and response generation
- Chunk overlap to ensure context preservation
- Response time and cost considerations
- Fine-tuning of under-the-hood prompts

### Human Component

The human component was considered in the design:
- User-friendly interface to facilitate easy interaction
- Clear instructions and examples provided within the interface
- Regular user studies to understand user needs and preferences

### AI Component

The AI component of the chatbot includes:
- Use of advanced LLMs for natural language understanding and generation
- Implementation of RAG system for improved response accuracy
- Vector similarity search for efficient information retrieval

### Classroom Implementation

Plans for implementing the chatbot in classrooms include:
- Pilot testing in Dr. Lohani's classroom
- Gathering feedback from students on usability and effectiveness
- Iterative improvements based on classroom use cases

### Future Considerations

Some questions and considerations for future development:
- How to effectively implement the chatbot in various classroom settings?
- Should users be given the option to choose the AI model?
- How to continually expand the knowledge base while maintaining performance?

## RESULTS AND DISCUSSION

The LEWAS chatbot has demonstrated several key achievements:
- Successful handling of queries about LEWAS operations, research, and environmental data
- Improved accessibility to lab information
- Streamlined knowledge transfer to new students
- Demonstrated ability to handle 100+ simultaneous users
- Reduced time for information retrieval compared to manual searching

The chatbot's performance highlights the potential of AI-assisted systems in environmental monitoring and education. It addresses the challenges of information overload and knowledge transfer in interdisciplinary research environments.

## CONCLUSION

The development of the AI-assisted chatbot for the LEWAS lab represents a significant advancement in managing and disseminating complex environmental monitoring information. By leveraging cutting-edge AI technologies and cloud computing resources, the chatbot provides an efficient, scalable, and user-friendly solution for accessing LEWAS data and research. This project demonstrates the potential of AI in enhancing environmental education and research, paving the way for similar applications in related fields.

## ACKNOWLEDGEMENT

This research is supported by NSF-IUSE grant #1915268. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

## REFERENCES

