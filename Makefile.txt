# Makefile for Python project setup

# Use python3.12 specifically. If this command isn't available,
# you might need to use 'python3' and ensure it points to 3.12.
PYTHON_EXE = python3.12
VENV_DIR = .venv
REQUIREMENTS = requirements.txt

# Phony targets don't represent files
.PHONY: setup clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup    - Create virtual environment '.venv' using $(PYTHON_EXE) and install requirements"
	@echo "  clean    - Remove the virtual environment directory"
	@echo "  help     - Show this help message"

# Setup target: Create venv and install dependencies
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: $(REQUIREMENTS)
	@echo "Creating virtual environment using $(PYTHON_EXE)..."
	$(PYTHON_EXE) -m venv $(VENV_DIR)
	@echo "Installing packages from $(REQUIREMENTS)..."
	$(VENV_DIR)/bin/python -m pip install --upgrade pip
	$(VENV_DIR)/bin/python -m pip install -r $(REQUIREMENTS)
	@echo ""
	@echo "Setup complete!"
	@echo "Activate the environment using: source $(VENV_DIR)/bin/activate"
	@# On Windows use: .venv\Scripts\activate
	@touch $(VENV_DIR)/bin/activate  # Mark setup as done

# Clean target: Remove the virtual environment
clean:
	@echo "Removing virtual environment $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
	@echo "Clean complete."