#!/bin/bash

# Kubernetes Local Development Cleanup Script
# Cleans up kind, minikube, and k3s clusters and associated resources

set -euo pipefail

CLUSTER_TYPE=${1:-kind}
CLUSTER_NAME="roleplay-dev"
REGISTRY_NAME="kind-registry"

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

cleanup_kind() {
    log_info "Cleaning up Kind cluster and resources..."
    
    # Delete kind cluster
    if kind get clusters 2>/dev/null | grep -q ${CLUSTER_NAME}; then
        log_info "Deleting kind cluster ${CLUSTER_NAME}..."
        kind delete cluster --name ${CLUSTER_NAME}
        log_success "Kind cluster ${CLUSTER_NAME} deleted"
    else
        log_warning "Kind cluster ${CLUSTER_NAME} does not exist"
    fi
    
    # Remove local registry
    if docker ps -a | grep -q ${REGISTRY_NAME}; then
        log_info "Removing local Docker registry..."
        docker stop ${REGISTRY_NAME} 2>/dev/null || true
        docker rm ${REGISTRY_NAME} 2>/dev/null || true
        log_success "Local registry removed"
    else
        log_warning "Local registry ${REGISTRY_NAME} does not exist"
    fi
    
    # Remove kind network if it exists and has no containers
    if docker network ls | grep -q "^[[:space:]]*[[:alnum:]]*[[:space:]]*kind"; then
        log_info "Checking kind network..."
        # Check if any containers are still connected to the kind network
        network_containers=$(docker network inspect kind --format='{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || echo "")
        if [ -z "$network_containers" ]; then
            log_info "Removing kind network..."
            docker network rm kind 2>/dev/null || true
            log_success "Kind network removed"
        else
            log_warning "Kind network still has connected containers: $network_containers"
            log_info "Disconnecting containers from kind network..."
            for container in $network_containers; do
                docker network disconnect kind $container 2>/dev/null || true
            done
            docker network rm kind 2>/dev/null || true
            log_success "Kind network cleaned up"
        fi
    else
        log_warning "Kind network does not exist"
    fi
    
    log_success "Kind cleanup completed"
}

cleanup_minikube() {
    log_info "Cleaning up Minikube cluster..."
    
    # Delete minikube cluster
    if minikube profile list 2>/dev/null | grep -q ${CLUSTER_NAME}; then
        log_info "Deleting minikube cluster ${CLUSTER_NAME}..."
        minikube delete -p ${CLUSTER_NAME}
        log_success "Minikube cluster ${CLUSTER_NAME} deleted"
    else
        log_warning "Minikube cluster ${CLUSTER_NAME} does not exist"
    fi
    
    log_success "Minikube cleanup completed"
}

cleanup_k3s() {
    log_info "Cleaning up K3s cluster..."
    
    # Stop and uninstall k3s
    if command -v k3s &> /dev/null; then
        log_info "Stopping K3s service..."
        sudo systemctl stop k3s 2>/dev/null || true
        
        log_info "Uninstalling K3s..."
        if [ -f /usr/local/bin/k3s-uninstall.sh ]; then
            sudo /usr/local/bin/k3s-uninstall.sh
        fi
        log_success "K3s uninstalled"
    else
        log_warning "K3s is not installed"
    fi
    
    log_success "K3s cleanup completed"
}

cleanup_docker_resources() {
    log_info "Cleaning up related Docker resources..."
    
    # Remove roleplay-platform related images
    images=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep roleplay-platform || true)
    if [ ! -z "$images" ]; then
        log_info "Removing roleplay-platform Docker images..."
        echo "$images" | xargs docker rmi -f 2>/dev/null || true
    fi
    
    # Clean up dangling images and volumes
    log_info "Cleaning up dangling Docker resources..."
    docker system prune -f --volumes 2>/dev/null || true
    
    log_success "Docker cleanup completed"
}

cleanup_kubectl_context() {
    log_info "Cleaning up kubectl contexts..."
    
    case "${CLUSTER_TYPE}" in
        "kind")
            context_name="kind-${CLUSTER_NAME}"
            ;;
        "minikube")
            context_name="${CLUSTER_NAME}"
            ;;
        "k3s")
            context_name="default"
            ;;
        *)
            log_warning "Unknown cluster type for context cleanup"
            return
            ;;
    esac
    
    # Remove kubectl context if it exists
    if kubectl config get-contexts -o name 2>/dev/null | grep -q "^${context_name}$"; then
        log_info "Removing kubectl context ${context_name}..."
        kubectl config delete-context ${context_name} 2>/dev/null || true
        log_success "kubectl context removed"
    else
        log_warning "kubectl context ${context_name} does not exist"
    fi
}

main() {
    log_info "Starting Kubernetes cleanup for ${CLUSTER_TYPE}..."
    
    case "${CLUSTER_TYPE}" in
        "kind")
            cleanup_kind
            ;;
        "minikube")
            cleanup_minikube
            ;;
        "k3s")
            cleanup_k3s
            ;;
        "all")
            log_info "Cleaning up all cluster types..."
            cleanup_kind
            cleanup_minikube
            cleanup_k3s
            ;;
        *)
            log_error "Unsupported cluster type: ${CLUSTER_TYPE}"
            log_info "Supported types: kind, minikube, k3s, all"
            exit 1
            ;;
    esac
    
    cleanup_kubectl_context
    cleanup_docker_resources
    
    log_success "Kubernetes cleanup completed successfully!"
    log_info "All ${CLUSTER_TYPE} cluster resources have been removed"
}

main "$@"