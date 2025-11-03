# Use an official Python runtime as a parent image
FROM python:3.12.3

# Set the working directory to /app
WORKDIR /app

# Install Jupyter Notebook
RUN pip install jupyter
RUN pip install pandas
RUN pip install numpy
RUN pip install shapely
RUN pip install ipywidgets
RUN pip install ipyleaflet

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./*.csv /app
COPY ./*.geojson /app
COPY ./*.ipynb /app
COPY ./*.json /app
COPY ./*.py /app

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]