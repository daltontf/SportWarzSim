if [ ! -f /.dockerenv ]; then
    echo "ERROR: This script must be run inside a Docker container."
    exit 1
fi

dnf install -y sqlite-devel cmake 
curl https://sh.rustup.rs -sSf | sh -s -- -y 
source $HOME/.cargo/env 
cd rust_lambda
cargo build --release