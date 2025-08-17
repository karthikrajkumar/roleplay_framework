# Kubernetes Native Development Setup

This guide walks you through setting up the AI Roleplay Platform using Kubernetes for local development.

## Prerequisites

Before starting, ensure you have the following installed:

- **Docker**: For building container images
- **kubectl**: Kubernetes command-line tool
- **One of the following local Kubernetes solutions**:
  - **Kind** (Kubernetes in Docker) - Recommended
  - **Minikube** - Alternative option
  - **k3s** - Lightweight option

## Quick Start

1. **Setup Kubernetes Cluster**
   ```bash
   make k8s-setup
   ```

2. **Deploy Complete Development Environment**
   ```bash
   make k8s-dev
   ```

3. **Check Health**
   ```bash
   make k8s-health
   ```

That's it! Your services will be available at:
- API Gateway: http://localhost:30000
- Documentation: http://localhost:30000/docs
- User Management: http://localhost:30001
- AI Orchestration: http://localhost:30003
- Grafana: http://localhost:30010

## Detailed Setup

### 1. Choose Your Kubernetes Platform

#### Option A: Kind (Recommended)
```bash
# Install kind if not already installed
go install sigs.k8s.io/kind@latest

# Setup with kind
K8S_CLUSTER_TYPE=kind make k8s-setup
```

#### Option B: Minikube
```bash
# Install minikube if not already installed
# See: https://minikube.sigs.k8s.io/docs/start/

# Setup with minikube
K8S_CLUSTER_TYPE=minikube make k8s-setup
```

#### Option C: k3s
```bash
# Setup with k3s (will install k3s)
K8S_CLUSTER_TYPE=k3s make k8s-setup
```

### 2. Environment Configuration

Create a `.env` file from the example:
```bash
make setup-env
```

Add your AI API keys to the `.env` file:
```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 3. Step-by-Step Deployment

If you prefer manual control over the deployment process:

```bash
# 1. Create namespace
make k8s-namespace

# 2. Create secrets (will prompt for API keys if not set)
make k8s-secrets

# 3. Build container images
make k8s-build

# 4. Deploy to Kubernetes
make k8s-deploy-dev

# 5. Initialize database
make k8s-db-init

# 6. Check health
make k8s-health
```

## Available Commands

### Primary Kubernetes Commands

| Command | Description |
|---------|-------------|
| `make k8s-dev` | Complete development setup |
| `make k8s-build` | Build and push all container images |
| `make k8s-deploy-dev` | Deploy development manifests |
| `make k8s-health` | Comprehensive health check |
| `make k8s-logs` | View logs from all services |
| `make k8s-status` | Show Kubernetes resource status |
| `make k8s-cleanup` | Clean up all resources |

### Service Management

| Command | Description |
|---------|-------------|
| `make k8s-restart` | Restart all services |
| `make k8s-restart-service SERVICE=api-gateway` | Restart specific service |
| `make k8s-logs-service SERVICE=api-gateway` | View logs for specific service |
| `make k8s-shell SERVICE=api-gateway` | Open shell in service pod |

### Database Operations

| Command | Description |
|---------|-------------|
| `make k8s-db-init` | Initialize database with migrations |
| `make k8s-db-migrate MSG="migration message"` | Create new migration |
| `make k8s-db-upgrade` | Apply database migrations |
| `make k8s-db-reset` | Reset database (destructive) |

### Port Forwarding

For direct access to services without NodePort:
```bash
make k8s-port-forward SERVICE=postgres PORT=5432
make k8s-port-forward SERVICE=redis PORT=6379
```

## Configuration

### Environment Variables

You can customize the deployment using environment variables:

```bash
# Kubernetes cluster type (kind, minikube, k3s)
export K8S_CLUSTER_TYPE=kind

# Docker registry for images
export REGISTRY=localhost:5001

# Image tag
export TAG=dev-$(git rev-parse --short HEAD)

# Base port for NodePort services
export K8S_NODE_PORT_BASE=30000
```

### Development vs Production

- **Development**: Uses simplified manifests with minimal resource requirements
- **Production**: Uses the full distributed architecture with clustering and high availability

Deploy production manifests:
```bash
make k8s-deploy  # Uses infrastructure/kubernetes/distributed-architecture.yaml
```

## Troubleshooting

### Common Issues

1. **Images not found**
   ```bash
   # Check if local registry is running (for kind)
   docker ps | grep registry
   
   # Rebuild images
   make k8s-build
   ```

2. **Services not ready**
   ```bash
   # Check pod status
   kubectl get pods -n roleplay-platform
   
   # View pod logs
   kubectl logs -n roleplay-platform deployment/api-gateway
   
   # Describe pod for detailed events
   kubectl describe pod -n roleplay-platform -l app=api-gateway
   ```

3. **Database connection issues**
   ```bash
   # Check if postgres is running
   kubectl get pods -n roleplay-platform -l app=postgres
   
   # Check database logs
   make k8s-logs-service SERVICE=postgres
   
   # Reinitialize database
   make k8s-db-reset
   make k8s-db-init
   ```

4. **Port conflicts**
   ```bash
   # Change base port
   export K8S_NODE_PORT_BASE=31000
   make k8s-deploy-dev
   ```

### Debugging Commands

```bash
# Get comprehensive system status
make k8s-status

# Check resource usage (if metrics-server is available)
kubectl top pods -n roleplay-platform

# View events
kubectl get events -n roleplay-platform --sort-by='.lastTimestamp'

# Check service endpoints
kubectl get endpoints -n roleplay-platform

# View secrets (masked)
kubectl get secrets -n roleplay-platform
```

### Health Check Interpretation

The health check script provides detailed information:

- ✓ Green: Service is healthy
- ⚠ Yellow: Service has warnings or is starting
- ✗ Red: Service has failed
- ? Unknown: Service status unclear

### Performance Tuning

For better performance on resource-constrained systems:

1. **Reduce replica counts**:
   Edit the development manifests to use fewer replicas

2. **Adjust resource limits**:
   Modify resource requests/limits in the manifests

3. **Use resource quotas**:
   ```bash
   kubectl create quota dev-quota --hard=cpu=4,memory=8Gi -n roleplay-platform
   ```

## Service Architecture

The Kubernetes deployment includes:

### Core Services
- **API Gateway** (port 30000): Main entry point and routing
- **User Management** (port 30001): Authentication and user data
- **AI Orchestration** (port 30003): AI model coordination
- **Notification** (port 30004): Event notifications
- **Analytics** (port 30005): Usage analytics
- **Real-time Communication** (port 30006): WebSocket connections

### Infrastructure
- **PostgreSQL**: Primary database
- **Redis**: Caching and session storage
- **Grafana** (port 30010): Monitoring and dashboards

### Networking
- Services communicate via Kubernetes DNS
- External access via NodePort services
- Network policies provide security isolation

## Migration from Docker Compose

If migrating from docker-compose:

1. **Legacy commands still work**: `make dev`, `make up`, `make down` now use Kubernetes
2. **Data migration**: Export data from docker volumes before switching
3. **Configuration**: Environment variables in `.env` are automatically used
4. **Port mapping**: Services are now on NodePort ranges (30000+) instead of direct mapping

## Next Steps

1. **Set up CI/CD**: Integrate with your continuous integration pipeline
2. **Production deployment**: Use `make k8s-deploy` for production-ready manifests
3. **Monitoring**: Configure additional monitoring with Prometheus and Grafana
4. **Scaling**: Use `kubectl scale` to adjust replica counts based on load

For production deployments, see the distributed architecture in `infrastructure/kubernetes/distributed-architecture.yaml`.