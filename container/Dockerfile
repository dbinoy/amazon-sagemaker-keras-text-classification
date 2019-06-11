# Build an image that can do training and inference in SageMaker

FROM tensorflow-base:1.8.0-cpu-py2

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY sagemaker_keras_text_classification /opt/program
WORKDIR /opt/program