# Use an official Python runtime as a parent image
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    pkg-config \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory to /app
WORKDIR /build 

# Install only what is need for the Rust part
RUN pip3 install maturin
RUN pip3 install patchelf

COPY ./src/ ./src/
COPY ./*.toml .

# Install the current package in editable mode and build the Rust extension in release mode
RUN pip install -e . 
RUN maturin build --release

FROM python:3.12-slim

WORKDIR /app

# Copy only the built wheel
COPY --from=builder /build/target/wheels/*.whl /tmp/

# Install wheel
RUN pip install /tmp/pyrust-0.1.0-cp312-cp312-manylinux_2_39_x86_64.whl

# Install Python dependencies
RUN pip3 install jupyter
RUN pip3 install ipywidgets
RUN pip3 install ipyleaflet
RUN pip3 install itables
RUN pip3 install ipywidgets_bokeh
RUN pip3 install voila
RUN pip3 install polars

# Copy the current directory contents into the container at /app
COPY ./*.csv /app
COPY ./*.geojson /app
COPY ./*.ipynb /app
COPY ./*.json /app
COPY ./*.py /app
COPY ./*.png /app

# Make port 8888 available to the world outside this container
EXPOSE 8866

# Run Voila when the container launches
CMD ["voila", "--theme=dark", "--no-browser", "--Voila.ip=0.0.0.0", "--Voila.port=8866", "--Voila.allow_root=True"]