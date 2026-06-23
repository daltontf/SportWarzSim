
This started as a Jupyter notebook and turned into an exercise in scaling and distribution. 

Juypter notebooks were at first an exercise to implement the idea and sharpen my Python skills. Being able to host notebooks using the Voila library was worked very well as long as someone downloads the Docker image and runs it locally and preferrably a Linux machine. In theory it should work on a Windows and Mac/OS but users have to be tech savvy in all of these situations.

I want feedback from sports fans and people who like to speculate on such things. Those people are not necessarily techically inclined so I decided to just host the Voila application somewhere.

I eventually hosted it on an inexpensive Amazon EC2 instance. Using Python for the computation meant that one user can bring the instance to its knees running four notebooks at the same time. It wasn't ready for public consumption.

There was optimizations done to the Python code that speed things up but after attending a Python meetup on [PyO3](https://pyo3.rs), the obvious next step was Rust!

Writing the computations in Rust sped things up immenselessly, but I still dissatified that I was burning my free AWS credits on a idling EC2 instance. So the next step is a Rust-based computation as a REST endpoint for an AWS Lambda and with a React or Jupyter-Lite client.

- [`jupyter`](./jupyter/README.md)

The original project using PyO3 to interface with the Rust computation logic.

- [rust_calc](./rust_calc/README.md)

At first the Lambda, PyO3 and Tauri version duplicated the core computation logic. This keep that "DRY".

- [`rust_geogen`](./rust_geogen/README.md)

Rust project for transforming GeoJSON and CSV files for consumption by `rust_calc`. 

- [`rust_lambda`](./rust_lambda/README.md)

The Rust computation logic wrapped in a AWS Lambda accessible via REST.

- [`tauri_map`](./tauri_map/README.md)

Tauri application based on React.

- [`aws-sam`](./aws-sam/README.md)

AWS Serverless Application Model for deploying Lambda zip, Gateway API configuration and React frontend.