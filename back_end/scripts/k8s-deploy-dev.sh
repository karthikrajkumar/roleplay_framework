#!/bin/bash

# Kubernetes Development Deployment Script
# Creates simplified manifests for local development

set -euo pipefail

NAMESPACE=${1:-roleplay-platform}
REGISTRY=${2:-localhost:5001}
TAG=${3:-dev-latest}
NODE_PORT_BASE=${4:-30000}

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

# Create development manifests
create_dev_manifests() {
    log_info "Creating development Kubernetes manifests..."
    
    # Create temporary directory for manifests
    DEV_MANIFESTS_DIR="/tmp/k8s-dev-manifests"
    mkdir -p ${DEV_MANIFESTS_DIR}
    
    # PostgreSQL (simplified for development)
    cat > ${DEV_MANIFESTS_DIR}/postgres-dev.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: ${NAMESPACE}
  labels:
    app: postgres
    tier: database
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
        tier: database
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: roleplay_platform
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - roleplay_user
            - -d
            - roleplay_platform
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - roleplay_user
            - -d
            - roleplay_platform
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: ${NAMESPACE}
  labels:
    app: postgres
spec:
  ports:
  - port: 5432
    targetPort: 5432
  selector:
    app: postgres
  type: ClusterIP
EOF

    # Redis (simplified for development)
    cat > ${DEV_MANIFESTS_DIR}/redis-dev.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ${NAMESPACE}
  labels:
    app: redis
    tier: cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
        tier: cache
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --appendonly
        - "yes"
        - --requirepass
        - "\$(REDIS_PASSWORD)"
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "125m"
          limits:
            memory: "512Mi"
            cpu: "250m"
      volumes:
      - name: redis-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: ${NAMESPACE}
  labels:
    app: redis
spec:
  ports:
  - port: 6379
    targetPort: 6379
  selector:
    app: redis
  type: ClusterIP
EOF

    # API Gateway
    create_service_manifest "api-gateway" 8000 0
    
    # User Management
    create_service_manifest "user-management" 8001 1
    
    # AI Orchestration  
    create_service_manifest "ai-orchestration" 8003 3
    
    # Notification Service
    create_service_manifest "notification" 8004 4
    
    # Analytics Service
    create_service_manifest "analytics" 8005 5
    
    # Real-time Communication
    create_service_manifest "real-time-communication" 8006 6
    
    # Monitoring (Grafana)
    cat > ${DEV_MANIFESTS_DIR}/monitoring-dev.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: ${NAMESPACE}
  labels:
    app: grafana
    tier: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
        tier: monitoring
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: monitoring-secret
              key: grafana-admin-password
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        resources:
          requests:
            memory: "256Mi"
            cpu: "125m"
          limits:
            memory: "512Mi"
            cpu: "250m"
      volumes:
      - name: grafana-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: ${NAMESPACE}
  labels:
    app: grafana
spec:
  ports:
  - port: 3000
    targetPort: 3000
    nodePort: $((NODE_PORT_BASE + 10))
  selector:
    app: grafana
  type: NodePort
EOF

    log_success "Development manifests created in ${DEV_MANIFESTS_DIR}"
}

create_service_manifest() {
    local service_name=$1
    local port=$2
    local node_port_offset=$3
    
    cat > ${DEV_MANIFESTS_DIR}/${service_name}-dev.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${service_name}
  namespace: ${NAMESPACE}
  labels:
    app: ${service_name}
    tier: application
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${service_name}
  template:
    metadata:
      labels:
        app: ${service_name}
        tier: application
    spec:
      containers:
      - name: ${service_name}
        image: ${REGISTRY}/roleplay-platform-${service_name}:${TAG}
        ports:
        - containerPort: ${port}
        env:
        - name: DATABASE_URL
          value: "postgresql+asyncpg://roleplay_user:\$(POSTGRES_PASSWORD)@postgres:5432/roleplay_platform"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@redis:6379/0"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
EOF

    # Add AI-specific environment variables for AI orchestration service
    if [[ "${service_name}" == "ai-orchestration" ]]; then
        cat >> ${DEV_MANIFESTS_DIR}/${service_name}-dev.yaml <<EOF
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-secret
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-secret
              key: anthropic-api-key
EOF
    fi

    cat >> ${DEV_MANIFESTS_DIR}/${service_name}-dev.yaml <<EOF
        resources:
          requests:
            memory: "256Mi"
            cpu: "125m"
          limits:
            memory: "512Mi"
            cpu: "250m"
        livenessProbe:
          httpGet:
            path: /health
            port: ${port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: ${port}
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ${service_name}
  namespace: ${NAMESPACE}
  labels:
    app: ${service_name}
spec:
  ports:
  - port: ${port}
    targetPort: ${port}
    nodePort: $((NODE_PORT_BASE + node_port_offset))
  selector:
    app: ${service_name}
  type: NodePort
EOF
}

deploy_manifests() {
    log_info "Deploying development manifests to Kubernetes..."
    
    # Apply infrastructure first
    kubectl apply -f ${DEV_MANIFESTS_DIR}/postgres-dev.yaml
    kubectl apply -f ${DEV_MANIFESTS_DIR}/redis-dev.yaml
    
    # Wait for infrastructure to be ready
    log_info "Waiting for infrastructure to be ready..."
    kubectl wait --for=condition=available deployment/postgres -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/redis -n ${NAMESPACE} --timeout=300s
    
    # Apply application services
    for manifest in ${DEV_MANIFESTS_DIR}/*-dev.yaml; do
        if [[ "$(basename ${manifest})" != "postgres-dev.yaml" ]] && [[ "$(basename ${manifest})" != "redis-dev.yaml" ]]; then
            log_info "Applying $(basename ${manifest})..."
            kubectl apply -f ${manifest}
        fi
    done
    
    log_success "All manifests deployed successfully"
}

show_access_info() {
    log_info "Development environment access information:"
    echo ""
    echo "Services are accessible via NodePort:"
    echo "  🌐 API Gateway:     http://localhost:$((NODE_PORT_BASE + 0))"
    echo "  📚 Documentation:   http://localhost:$((NODE_PORT_BASE + 0))/docs"
    echo "  👤 User Management: http://localhost:$((NODE_PORT_BASE + 1))"
    echo "  🤖 AI Orchestration: http://localhost:$((NODE_PORT_BASE + 3))"
    echo "  📢 Notifications:   http://localhost:$((NODE_PORT_BASE + 4))"
    echo "  📊 Analytics:       http://localhost:$((NODE_PORT_BASE + 5))"
    echo "  💬 Real-time:       http://localhost:$((NODE_PORT_BASE + 6))"
    echo "  📈 Grafana:         http://localhost:$((NODE_PORT_BASE + 10))"
    echo ""
    echo "Database access (port-forward required):"
    echo "  🐘 PostgreSQL: kubectl port-forward -n ${NAMESPACE} svc/postgres 5432:5432"
    echo "  🔴 Redis:      kubectl port-forward -n ${NAMESPACE} svc/redis 6379:6379"
    echo ""
    echo "Useful commands:"
    echo "  📊 Status:  make k8s-status"
    echo "  📜 Logs:    make k8s-logs"
    echo "  🏥 Health:  make k8s-health"
}

main() {
    log_info "Starting development deployment to namespace: ${NAMESPACE}"
    
    create_dev_manifests
    deploy_manifests
    show_access_info
    
    log_success "Development deployment completed successfully!"
}

main "$@"