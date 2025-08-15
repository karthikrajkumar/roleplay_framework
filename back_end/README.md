# AI Roleplay Platform Backend

A robust, scalable microservices architecture for an advanced AI roleplay platform built with FastAPI, following SOLID principles and clean architecture patterns.

## 🏗️ Architecture Overview

The backend is designed as a microservices architecture with the following components:

### Core Services
- **API Gateway** (`8000`) - Request routing, authentication, rate limiting
- **User Management** (`8001`) - User authentication, profiles, preferences
- **Roleplay Service** (`8002`) - Session management, characters, scenarios
- **AI Orchestration** (`8003`) - Multi-provider AI integration and routing
- **Notification Service** (`8004`) - Real-time notifications and alerts
- **Analytics Service** (`8005`) - Usage tracking and insights
- **Real-time Communication** (`8006`) - WebSocket and WebRTC support

### Infrastructure Components
- **PostgreSQL** - Primary database
- **Redis** - Caching and session storage
- **Qdrant** - Vector database for embeddings
- **RabbitMQ** - Message queuing
- **Consul** - Service discovery
- **Prometheus/Grafana** - Monitoring and metrics

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Make (optional, for convenience commands)

### Development Setup

1. **Clone and navigate to the backend directory**
   ```bash
   git clone <repository>
   cd back_end
   ```

2. **Setup environment and start services**
   ```bash
   make dev
   ```

   This command will:
   - Copy `.env.example` to `.env`
   - Build all Docker images
   - Start all services
   - Initialize the database

3. **Verify services are running**
   ```bash
   make health
   ```

4. **Access the services**
   - API Gateway: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Grafana Dashboard: http://localhost:3000 (admin/admin)
   - RabbitMQ Management: http://localhost:15672 (guest/guest)

## 📁 Project Structure

```
back_end/
├── shared/                          # Shared library
│   ├── domain/
│   │   └── entities/               # Domain entities (User, Session, etc.)
│   ├── interfaces/                 # Repository and service interfaces
│   └── utils/                      # Common utilities
├── services/
│   ├── api-gateway/                # API Gateway service
│   ├── user-management/            # User service
│   ├── roleplay-service/          # Roleplay management
│   ├── ai-orchestration/          # AI provider integration
│   ├── notification-service/      # Notifications
│   ├── analytics-service/         # Analytics and tracking
│   └── real-time-communication/   # WebSocket/WebRTC
├── infrastructure/                 # Infrastructure configs
│   ├── kubernetes/                # K8s manifests
│   ├── monitoring/               # Prometheus/Grafana configs
│   └── terraform/               # Infrastructure as code
├── docker-compose.yml            # Docker services definition
├── Makefile                     # Development commands
└── README.md                   # This file
```

## 🔧 Development Commands

The project includes a comprehensive Makefile with common development tasks:

### Service Management
```bash
make up              # Start all services
make down            # Stop all services
make restart         # Restart all services
make logs            # View logs from all services
make health          # Check service health
```

### Development Workflow
```bash
make dev             # Complete development setup
make build           # Build Docker images
make db-init         # Initialize database
make db-migrate      # Create migration
make test            # Run tests
```

### Code Quality
```bash
make format          # Format code with black/isort
make lint            # Run linting
make security        # Run security checks
make test-coverage   # Run tests with coverage
```

## 🛠️ Service Architecture

### Domain-Driven Design
Each service follows DDD principles:
- **Domain Layer** - Core business entities and rules
- **Application Layer** - Use cases and service orchestration  
- **Infrastructure Layer** - Database, external services, messaging
- **Presentation Layer** - HTTP endpoints and API contracts

### Repository Pattern
All data access is abstracted through repository interfaces:
```python
from shared.interfaces.repository import IUserRepository

class UserService:
    def __init__(self, user_repository: IUserRepository):
        self.user_repository = user_repository
```

### Dependency Injection
Services use a lightweight DI container for loose coupling:
```python
from shared.utils.dependency_injection import Container, inject

Container.register_singleton(IUserRepository, UserRepository)

@inject
async def endpoint(user_service: IUserService):
    # Dependencies automatically injected
    pass
```

## 🔐 Authentication & Security

### JWT-based Authentication
- Access tokens (30 minutes)
- Refresh tokens (30 days) 
- Role-based access control (RBAC)
- Subscription tier validation

### Security Features
- Password hashing with bcrypt
- Rate limiting per IP/user
- Request ID tracking
- Input validation and sanitization
- SQL injection protection via SQLAlchemy ORM

## 🤖 AI Integration

### Multi-Provider Support
The AI Orchestration service supports multiple providers:
- **OpenAI** (GPT-3.5, GPT-4)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **Cohere** 
- **Local models** (Ollama, etc.)

### Features
- Automatic model selection based on context
- Cost estimation and tracking
- Response caching
- Streaming support
- Circuit breaker for reliability

## 📊 Database Design

### User Management
- Users, profiles, preferences
- Authentication tokens
- Session tracking
- Audit logging

### Roleplay Data
- Sessions, messages, participants
- Characters with AI personalities
- Scenarios and templates
- Usage analytics

### AI Data
- Provider and model configurations
- Response caching and analytics
- Vector embeddings for similarity search

## 🚀 Deployment

### Docker Compose (Development)
```bash
docker-compose up -d
```

### Kubernetes (Production)
```bash
kubectl apply -f infrastructure/kubernetes/
```

### Environment Variables
Copy `.env.example` to `.env` and configure:
- Database URLs
- AI provider API keys
- Redis/cache settings
- SMTP configuration
- File storage settings

## 📈 Monitoring & Observability

### Metrics & Monitoring
- **Prometheus** - Metrics collection
- **Grafana** - Dashboards and visualization
- **Structured logging** - JSON logs with correlation IDs
- **Health checks** - Kubernetes-compatible health endpoints

### Key Metrics
- Request latency and throughput
- Error rates by service
- Database connection pools
- AI provider response times
- User engagement metrics

## 🧪 Testing

### Test Structure
Each service includes:
- **Unit tests** - Business logic and utilities
- **Integration tests** - Database and external services
- **API tests** - Endpoint behavior and contracts

### Running Tests
```bash
make test                    # All tests
make test-service SERVICE=user-management  # Specific service
make test-coverage          # With coverage reports
```

## 🔄 Real-time Features

### WebSocket Support
- Live roleplay sessions
- Real-time notifications
- User presence tracking
- Session participant management

### WebRTC Integration
- Voice/video calls during roleplay
- Screen sharing capabilities
- Peer-to-peer communication
- TURN server integration

## 📝 API Documentation

### Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Versioning
All APIs use versioned endpoints:
```
/api/v1/users/
/api/v1/roleplay/sessions/
/api/v1/ai/models/
```

## 🤝 Contributing

1. **Follow the established patterns**
   - Repository pattern for data access
   - Service layer for business logic
   - Dependency injection for loose coupling

2. **Code Quality Standards**
   - Type hints for all functions
   - Comprehensive docstrings
   - 90%+ test coverage
   - Security-first approach

3. **Development Workflow**
   ```bash
   # Create feature branch
   git checkout -b feature/new-feature
   
   # Make changes and test
   make test
   make lint
   make security
   
   # Commit and push
   git commit -m "feat: add new feature"
   git push origin feature/new-feature
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   make logs              # Check service logs
   make health           # Check service health
   docker-compose ps     # Check container status
   ```

2. **Database connection issues**
   ```bash
   make db-reset         # Reset database
   docker-compose restart postgres
   ```

3. **Port conflicts**
   ```bash
   # Change ports in docker-compose.yml
   # Update .env file accordingly
   ```

### Performance Issues
- Check database query performance with `EXPLAIN ANALYZE`
- Monitor Redis memory usage and hit rates
- Review service metrics in Grafana dashboards
- Analyze API response times in logs

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/14/orm/extensions/asyncio.html)
- [Docker Compose](https://docs.docker.com/compose/)
- [Kubernetes](https://kubernetes.io/docs/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.