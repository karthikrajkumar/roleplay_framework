#!/bin/bash

# Kubernetes Local Development Setup Script
# Supports kind, minikube, and k3s

set -euo pipefail

CLUSTER_TYPE=${1:-kind}
CLUSTER_NAME="roleplay-dev"
REGISTRY_NAME="kind-registry"
REGISTRY_PORT="5001"

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

setup_kind() {
    log_info "Setting up Kind cluster..."
    
    # Check if kind is installed
    if ! command -v kind &> /dev/null; then
        log_error "Kind is not installed. Installing kind..."
        # Install kind using Go (requires Go to be installed)
        if command -v go &> /dev/null; then
            go install sigs.k8s.io/kind@latest
        else
            log_error "Go is not installed. Please install kind manually: https://kind.sigs.k8s.io/docs/user/quick-start/"
            exit 1
        fi
    fi
    
    # Create local registry if it doesn't exist
    if ! docker ps | grep -q ${REGISTRY_NAME}; then
        log_info "Creating local Docker registry..."
        docker run -d --restart=always -p "127.0.0.1:${REGISTRY_PORT}:5000" --name "${REGISTRY_NAME}" registry:2
    fi
    
    # Create kind cluster config
    cat > /tmp/kind-config.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors."localhost:${REGISTRY_PORT}"]
    endpoint = ["http://${REGISTRY_NAME}:5000"]
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
  - containerPort: 30000
    hostPort: 30000
    protocol: TCP
  - containerPort: 30001
    hostPort: 30001
    protocol: TCP
  - containerPort: 30002
    hostPort: 30002
    protocol: TCP
  - containerPort: 30003
    hostPort: 30003
    protocol: TCP
  - containerPort: 30004
    hostPort: 30004
    protocol: TCP
  - containerPort: 30005
    hostPort: 30005
    protocol: TCP
  - containerPort: 30010
    hostPort: 30010
    protocol: TCP
EOF
    
    # Create kind cluster
    if kind get clusters | grep -q ${CLUSTER_NAME}; then
        log_warning "Kind cluster ${CLUSTER_NAME} already exists"
    else
        log_info "Creating kind cluster ${CLUSTER_NAME}..."
        kind create cluster --name ${CLUSTER_NAME} --config /tmp/kind-config.yaml
    fi
    
    # Connect registry to cluster network
    # Note: kind create cluster automatically creates the 'kind' network
    # We just need to connect our registry to it
    docker network connect "kind" "${REGISTRY_NAME}" 2>/dev/null || true
    
    # Set kubectl context
    kubectl cluster-info --context kind-${CLUSTER_NAME}
    
    log_success "Kind cluster setup completed"
}

setup_minikube() {
    log_info "Setting up Minikube cluster..."
    
    # Check if minikube is installed
    if ! command -v minikube &> /dev/null; then
        log_error "Minikube is not installed. Please install minikube first."
        exit 1
    fi
    
    # Start minikube
    if minikube status -p ${CLUSTER_NAME} | grep -q "Running"; then
        log_warning "Minikube cluster ${CLUSTER_NAME} is already running"
    else
        log_info "Starting minikube cluster ${CLUSTER_NAME}..."
        minikube start -p ${CLUSTER_NAME} --cpus=4 --memory=8192 --disk-size=50g
    fi
    
    # Enable addons
    minikube addons enable ingress -p ${CLUSTER_NAME}
    minikube addons enable registry -p ${CLUSTER_NAME}
    minikube addons enable metrics-server -p ${CLUSTER_NAME}
    
    # Set kubectl context
    kubectl config use-context ${CLUSTER_NAME}
    
    log_success "Minikube cluster setup completed"
}

setup_k3s() {
    log_info "Setting up K3s cluster..."
    
    # Check if k3s is installed
    if ! command -v k3s &> /dev/null; then
        log_info "Installing K3s..."
        curl -sfL https://get.k3s.io | sh -
        
        # Copy kubeconfig
        sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
        sudo chown $(whoami):$(whoami) ~/.kube/config
    else
        log_warning "K3s is already installed"
    fi
    
    # Verify k3s is running
    sudo systemctl status k3s
    
    log_success "K3s cluster setup completed"
}

install_ingress_controller() {
    if [[ "${CLUSTER_TYPE}" == "kind" ]]; then
        log_info "Installing NGINX Ingress Controller for Kind..."
        if kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml; then
            log_info "Waiting for ingress controller to be ready..."
            # Wait for ingress controller to be ready with better error handling
            if kubectl wait --namespace ingress-nginx \
                --for=condition=ready pod \
                --selector=app.kubernetes.io/component=controller \
                --timeout=300s 2>/dev/null; then
                log_success "NGINX Ingress Controller is ready"
            else
                log_warning "Ingress controller pods not ready yet, but continuing..."
                log_warning "You can check status later with: kubectl get pods -n ingress-nginx"
            fi
        else
            log_warning "Failed to install NGINX Ingress Controller, but continuing..."
            log_warning "NodePort services will still work for development"
        fi
    fi
}

main() {
    log_info "Starting Kubernetes setup for ${CLUSTER_TYPE}..."
    
    check_prerequisites
    
    case "${CLUSTER_TYPE}" in
        "kind")
            setup_kind
            install_ingress_controller
            ;;
        "minikube")
            setup_minikube
            ;;
        "k3s")
            setup_k3s
            ;;
        *)
            log_error "Unsupported cluster type: ${CLUSTER_TYPE}"
            log_info "Supported types: kind, minikube, k3s"
            exit 1
            ;;
    esac
    
    log_success "Kubernetes cluster setup completed successfully!"
    log_info "Cluster Type: ${CLUSTER_TYPE}"
    log_info "Cluster Name: ${CLUSTER_NAME}"
    
    if [[ "${CLUSTER_TYPE}" == "kind" ]]; then
        log_info "Local Registry: localhost:${REGISTRY_PORT}"
    fi
    
    # Show cluster info
    kubectl cluster-info
}

main "$@"