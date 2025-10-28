# Quickstart

This guide will help you get the CSAM-Prevention service up and running in a few simple steps.

## Prerequisites
- Python 3.8+
- Docker (optional)

## Development setup
To set up the development environment, clone the repository and use the `Makefile` to install the dependencies:
```bash
git clone https://github.com/Fayeblade1488/CSAM-Prevention.git
cd CSAM-Prevention
make setup
```

Once the setup is complete, you can run the tests to ensure everything is working correctly:
```bash
make test
```

## Production usage
To run the service in a production environment, you can use the `make run` command:
```bash
make run
```

Alternatively, you can build and run the service using Docker:
```bash
docker build -t csam-prevention .
docker run -p 8000:8000 csam-prevention
```
