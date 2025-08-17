#!/bin/bash

# Build and Push Container Images Script
# Builds all service images for local Kubernetes development

set -euo pipefail

REGISTRY=${1:-localhost:5001}
TAG=${2:-dev-latest}
SERVICES="api-gateway user-management ai-orchestration notification analytics real-time-communication"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Docker is available and running"
}

check_registry() {
    log_info "Checking if registry ${REGISTRY} is accessible..."
    
    # Try to ping the registry
    if curl -sf "http://${REGISTRY}/v2/" &>/dev/null; then
        log_success "Registry ${REGISTRY} is accessible"
    else
        log_warning "Registry ${REGISTRY} might not be accessible"
        log_warning "If using kind, make sure the local registry is running"
    fi
}

build_service() {
    local service=$1
    local service_dir="./services/${service}"
    local image_name="${REGISTRY}/roleplay-platform-${service}:${TAG}"
    
    log_info "Building ${service}..."
    
    if [[ ! -d "${service_dir}" ]]; then
        log_error "Service directory ${service_dir} not found"
        return 1
    fi
    
    if [[ ! -f "${service_dir}/Dockerfile" ]]; then
        log_warning "No Dockerfile found for ${service}, creating a basic one..."
        create_basic_dockerfile "${service_dir}"
    fi
    
    # Build the image from back_end directory with shared library context
    log_info "Building image: ${image_name}"
    cd /Users/k.rajkumar.kannan/Documents/research/role_play_framework/back_end
    if docker build -f "${service_dir}/Dockerfile" --build-arg SHARED_PATH=./shared -t "${image_name}" .; then
        log_success "Built ${image_name}"
    else
        log_error "Failed to build ${image_name}"
        return 1
    fi
    
    # Push the image
    log_info "Pushing image: ${image_name}"
    if docker push "${image_name}"; then
        log_success "Pushed ${image_name}"
    else
        log_error "Failed to push ${image_name}"
        return 1
    fi
}

create_basic_dockerfile() {
    local service_dir=$1
    local dockerfile="${service_dir}/Dockerfile"
    
    log_info "Creating basic Dockerfile for ${service_dir}"
    
    cat > "${dockerfile}" <<EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]
EOF
    
    log_success "Created basic Dockerfile at ${dockerfile}"
}

build_all_services() {
    local failed_services=()
    
    for service in ${SERVICES}; do
        log_info "Processing service: ${service}"
        
        if build_service "${service}"; then
            log_success "Successfully built and pushed ${service}"
        else
            log_error "Failed to build ${service}"
            failed_services+=("${service}")
        fi
        
        echo ""
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "All services built and pushed successfully!"
    else
        log_error "Failed to build the following services:"
        for service in "${failed_services[@]}"; do
            echo "  - ${service}"
        done
        exit 1
    fi
}

show_built_images() {
    log_info "Built images:"
    
    for service in ${SERVICES}; do
        local image_name="${REGISTRY}/roleplay-platform-${service}:${TAG}"
        if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "${image_name}"; then
            echo "  ✓ ${image_name}"
        else
            echo "  ✗ ${image_name} (not found)"
        fi
    done
}

main() {
    log_info "Building container images for Kubernetes deployment"
    log_info "Registry: ${REGISTRY}"
    log_info "Tag: ${TAG}"
    log_info "Services: ${SERVICES}"
    echo ""
    
    check_docker
    check_registry
    
    echo ""
    build_all_services
    
    echo ""
    show_built_images
    
    echo ""
    log_success "Build process completed!"
    log_info "You can now deploy with: make k8s-deploy-dev"
}

main "$@"