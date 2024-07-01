FROM public.ecr.aws/lambda/python:3.11
#3.12 doesnt work

#Copy the requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install SQLite3 to work with Chroma
# RUN pip install pysqlite3-binary

#Install the dependencies
RUN pip install -r requirements.txt --upgrade

#Local Testing  
EXPOSE 8000

#Set Environment Variable 
ENV IS_USING_IMAGE_RUNTIME=True

#Copy the source code
COPY src/* ${LAMBDA_TASK_ROOT}
COPY src/rag_app ${LAMBDA_TASK_ROOT}/rag_app
COPY src/data/chroma ${LAMBDA_TASK_ROOT}/data/chroma