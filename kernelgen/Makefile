# Makefile for KernelBench HIP Kernel Development
# Dynamic ROCm Environment Setup

SHELL := /bin/bash
PYTHON_VERSION := 3.12
VENV_DIR := .venv
REQUIREMENTS := requirements.txt

# Detect ROCm version
ROCM_VERSION := $(shell if [ -d /opt/rocm-7.0.0 ]; then echo "7.0"; elif [ -d /opt/rocm-6.4.1 ]; then echo "6.4"; else echo "unknown"; fi)
ROCM_PATH_DETECTED := $(shell if [ -d /opt/rocm-7.0.0 ]; then echo "/opt/rocm-7.0.0"; elif [ -d /opt/rocm-6.4.1 ]; then echo "/opt/rocm-6.4.1"; else echo "/opt/rocm"; fi)

# ROCm environment variables
export ROCM_PATH := $(ROCM_PATH_DETECTED)
export PYTORCH_ROCM_ARCH := gfx90a;gfx942
export CMAKE_PREFIX_PATH := $(ROCM_PATH_DETECTED):$(ROCM_PATH_DETECTED)/hip:/usr/local:/usr
export MAX_JOBS := 8
export HIP_FORCE_DEV_KERNARG := 1
export HSA_NO_SCRATCH_RECLAIM := 1
export HIPCC_COMPILE_FLAGS_APPEND := --offload-arch=gfx942
export AMDGPU_TARGETS := gfx942
export ROCM_ARCH := gfx942

.PHONY: help setup clean

help:
	@echo "KernelBench HIP Kernel Development - Makefile Commands"
	@echo "======================================================"
	@echo "make setup  - Complete environment setup (venv + deps)"
	@echo "make clean  - Remove virtual environment"

setup:
	@echo "Detected ROCm version: $(ROCM_VERSION) at $(ROCM_PATH_DETECTED)"
	@if [ "$(ROCM_VERSION)" = "unknown" ]; then \
		echo "ERROR: Could not detect ROCm installation"; \
		exit 1; \
	fi
	@echo "Creating virtual environment with uv..."
	@uv venv $(VENV_DIR) --python $(PYTHON_VERSION)
	@echo "✓ Virtual environment created"
	@echo "Installing PyTorch for ROCm $(ROCM_VERSION)..."
	@source $(VENV_DIR)/bin/activate && \
		uv pip install --upgrade pip setuptools wheel && \
		uv pip install setuptools==75.8.0 && \
		uv pip install setuptools_scm packaging && \
		if [ "$(ROCM_VERSION)" = "7.0" ]; then \
			uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0; \
		else \
			uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4; \
		fi
	@echo "✓ PyTorch installed"
	@echo "Installing Python dependencies..."
	@if [ ! -f $(REQUIREMENTS) ]; then \
		echo "Creating requirements.txt..."; \
		echo "# Core ML libraries" > $(REQUIREMENTS); \
		echo "torch" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# Build tools" >> $(REQUIREMENTS); \
		echo "ninja" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# LLM service dependencies" >> $(REQUIREMENTS); \
		echo "pyyaml" >> $(REQUIREMENTS); \
		echo "httpx" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# Utilities" >> $(REQUIREMENTS); \
		echo "numpy" >> $(REQUIREMENTS); \
	fi
	@source $(VENV_DIR)/bin/activate && uv pip install -r $(REQUIREMENTS)
	@echo "✓ Setup complete! Activate with: source $(VENV_DIR)/bin/activate"

clean:
	@echo "Removing virtual environment and build caches..."
	@rm -rf $(VENV_DIR)
	@find . -type d -name "build_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"


ACTIVATE_VENV_CMD = exec bash -c "source .venv/bin/activate && exec bash"
act:
	$(ACTIVATE_VENV_CMD) 