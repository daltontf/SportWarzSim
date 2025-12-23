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
RUN pip install voila
RUN pip install itables
RUN pip install panel
RUN pip install ipywidgets_bokeh

# Make port 8888 available to the world outside this container
EXPOSE 8866

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./*.csv /app
COPY ./*.geojson /app
COPY ./*.ipynb /app
COPY ./*.json /app
COPY ./*.py /app

# Run Voila when the container launches
CMD ["voila", "--theme=dark", "--no-browser", "--Voila.ip=0.0.0.0"]