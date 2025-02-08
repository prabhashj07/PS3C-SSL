# Makefile for downloading P3SC Classification dataset

# Default target: Help message 
help: 
	@echo "Usage make <target>"
	@echo "Targets:"
	@echo "dataset: Download P3SC dataset"

# Download data
dataset:
	@echo "Downloading the P3SC dataset....."
	@python scripts/download_dataset.py

# Run the train scripts
run:
	@python train.py 
