#!/bin/bash

# Kubernetes Secrets Creation Script
# Creates all necessary secrets for the roleplay platform

set -euo pipefail

NAMESPACE=${1:-roleplay-platform}

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

# Function to generate a random password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Function to create secret if it doesn't exist
create_secret_if_not_exists() {
    local secret_name=$1
    local secret_type=$2
    shift 2
    local secret_data=("$@")
    
    if kubectl get secret ${secret_name} -n ${NAMESPACE} &>/dev/null; then
        log_warning "Secret ${secret_name} already exists in namespace ${NAMESPACE}"
    else
        log_info "Creating secret ${secret_name}..."
        kubectl create secret ${secret_type} ${secret_name} -n ${NAMESPACE} "${secret_data[@]}"
        log_success "Secret ${secret_name} created successfully"
    fi
}

create_postgres_secret() {
    log_info "Creating PostgreSQL secrets..."
    
    # Generate passwords if not set
    POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(generate_password)}
    POSTGRES_REPLICATION_PASSWORD=${POSTGRES_REPLICATION_PASSWORD:-$(generate_password)}
    
    create_secret_if_not_exists "postgres-secret" "generic" \
        "--from-literal=password=${POSTGRES_PASSWORD}" \
        "--from-literal=replication-password=${POSTGRES_REPLICATION_PASSWORD}" \
        "--from-literal=username=roleplay_user" \
        "--from-literal=database=roleplay_platform"
    
    echo "PostgreSQL credentials:"
    echo "  Username: roleplay_user"
    echo "  Password: ${POSTGRES_PASSWORD}"
    echo "  Replication Password: ${POSTGRES_REPLICATION_PASSWORD}"
    echo ""
}

create_redis_secret() {
    log_info "Creating Redis secrets..."
    
    REDIS_PASSWORD=${REDIS_PASSWORD:-$(generate_password)}
    
    create_secret_if_not_exists "redis-secret" "generic" \
        "--from-literal=password=${REDIS_PASSWORD}"
    
    echo "Redis credentials:"
    echo "  Password: ${REDIS_PASSWORD}"
    echo ""
}

create_rabbitmq_secret() {
    log_info "Creating RabbitMQ secrets..."
    
    RABBITMQ_USER=${RABBITMQ_USER:-roleplay_user}
    RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD:-$(generate_password)}
    RABBITMQ_ERLANG_COOKIE=${RABBITMQ_ERLANG_COOKIE:-$(generate_password)}
    
    create_secret_if_not_exists "rabbitmq-secret" "generic" \
        "--from-literal=username=${RABBITMQ_USER}" \
        "--from-literal=password=${RABBITMQ_PASSWORD}" \
        "--from-literal=erlang-cookie=${RABBITMQ_ERLANG_COOKIE}"
    
    echo "RabbitMQ credentials:"
    echo "  Username: ${RABBITMQ_USER}"
    echo "  Password: ${RABBITMQ_PASSWORD}"
    echo "  Erlang Cookie: ${RABBITMQ_ERLANG_COOKIE}"
    echo ""
}

create_ai_api_secrets() {
    log_info "Creating AI API secrets..."
    
    # These should be provided as environment variables
    OPENAI_API_KEY=${OPENAI_API_KEY:-""}
    ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-""}
    
    if [[ -z "${OPENAI_API_KEY}" ]] && [[ -z "${ANTHROPIC_API_KEY}" ]]; then
        log_warning "No AI API keys provided. Please set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables"
        log_warning "Creating empty secret for now..."
        
        create_secret_if_not_exists "ai-api-secret" "generic" \
            "--from-literal=openai-api-key=" \
            "--from-literal=anthropic-api-key="
    else
        create_secret_if_not_exists "ai-api-secret" "generic" \
            "--from-literal=openai-api-key=${OPENAI_API_KEY}" \
            "--from-literal=anthropic-api-key=${ANTHROPIC_API_KEY}"
        
        echo "AI API Keys configured (showing only first 10 characters):"
        [[ -n "${OPENAI_API_KEY}" ]] && echo "  OpenAI: ${OPENAI_API_KEY:0:10}..."
        [[ -n "${ANTHROPIC_API_KEY}" ]] && echo "  Anthropic: ${ANTHROPIC_API_KEY:0:10}..."
    fi
    echo ""
}

create_tls_secret() {
    log_info "Creating TLS secrets for development..."
    
    # Create self-signed certificate for development
    CERT_DIR="/tmp/k8s-certs"
    mkdir -p ${CERT_DIR}
    
    if [[ ! -f "${CERT_DIR}/tls.crt" ]] || [[ ! -f "${CERT_DIR}/tls.key" ]]; then
        log_info "Generating self-signed certificate..."
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ${CERT_DIR}/tls.key \
            -out ${CERT_DIR}/tls.crt \
            -subj "/CN=roleplay-platform.local/O=roleplay-platform" \
            -addext "subjectAltName=DNS:roleplay-platform.local,DNS:*.roleplay-platform.local,DNS:localhost,IP:127.0.0.1"
    fi
    
    create_secret_if_not_exists "tls-secret" "tls" \
        "--cert=${CERT_DIR}/tls.crt" \
        "--key=${CERT_DIR}/tls.key"
    
    echo "TLS certificate created for development use"
    echo "  Subject: /CN=roleplay-platform.local/O=roleplay-platform"
    echo "  Valid for: roleplay-platform.local, *.roleplay-platform.local, localhost, 127.0.0.1"
    echo ""
}

create_monitoring_secrets() {
    log_info "Creating monitoring secrets..."
    
    GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-$(generate_password)}
    
    create_secret_if_not_exists "monitoring-secret" "generic" \
        "--from-literal=grafana-admin-password=${GRAFANA_ADMIN_PASSWORD}"
    
    echo "Monitoring credentials:"
    echo "  Grafana Admin Password: ${GRAFANA_ADMIN_PASSWORD}"
    echo ""
}

save_credentials() {
    log_info "Saving credentials to .env.k8s file..."
    
    cat > .env.k8s <<EOF
# Kubernetes Development Environment Credentials
# Generated on $(date)

# PostgreSQL
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_REPLICATION_PASSWORD=${POSTGRES_REPLICATION_PASSWORD}

# Redis
REDIS_PASSWORD=${REDIS_PASSWORD}

# RabbitMQ
RABBITMQ_USER=${RABBITMQ_USER}
RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD}
RABBITMQ_ERLANG_COOKIE=${RABBITMQ_ERLANG_COOKIE}

# AI APIs (set these manually)
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Monitoring
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}

# Kubernetes
NAMESPACE=${NAMESPACE}
EOF
    
    log_success "Credentials saved to .env.k8s file"
    log_warning "Please keep this file secure and do not commit it to version control"
}

main() {
    log_info "Creating Kubernetes secrets for namespace: ${NAMESPACE}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    create_postgres_secret
    create_redis_secret
    create_rabbitmq_secret
    create_ai_api_secrets
    create_tls_secret
    create_monitoring_secrets
    
    save_credentials
    
    log_success "All secrets created successfully!"
    log_info "To view secrets: kubectl get secrets -n ${NAMESPACE}"
    log_info "To update AI API keys later, run:"
    log_info "  kubectl patch secret ai-api-secret -n ${NAMESPACE} -p='{\"data\":{\"openai-api-key\":\"<base64-encoded-key>\"}}'"
}

main "$@"