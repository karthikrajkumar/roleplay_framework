"""
Service Discovery client for Consul integration.

This module provides functionality for service registration,
deregistration, and health checking with Consul.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Service information for registration."""
    name: str
    id: str
    address: str
    port: int
    tags: List[str]
    health_check_url: Optional[str] = None


class ServiceDiscovery:
    """Consul service discovery client."""
    
    def __init__(self, consul_url: str = "http://localhost:8500"):
        """Initialize service discovery client."""
        self.consul_url = consul_url.rstrip('/')
        self.client = httpx.AsyncClient()
        
    async def register_service(
        self,
        name: str,
        port: int,
        address: str = "localhost",
        health_check_url: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Register a service with Consul."""
        service_id = f"{name}-{address}-{port}"
        
        service_data = {
            "ID": service_id,
            "Name": name,
            "Address": address,
            "Port": port,
            "Tags": tags or []
        }
        
        # Add health check if provided
        if health_check_url:
            service_data["Check"] = {
                "HTTP": health_check_url,
                "Interval": "30s",
                "Timeout": "10s",
                "DeregisterCriticalServiceAfter": "90s"
            }
        
        try:
            response = await self.client.put(
                f"{self.consul_url}/v1/agent/service/register",
                json=service_data
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully registered service {name} with ID {service_id}")
                return True
            else:
                logger.error(f"Failed to register service {name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering service {name}: {e}")
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from Consul."""
        try:
            response = await self.client.put(
                f"{self.consul_url}/v1/agent/service/deregister/{service_id}"
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully deregistered service {service_id}")
                return True
            else:
                logger.error(f"Failed to deregister service {service_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error deregistering service {service_id}: {e}")
            return False
    
    async def discover_service(self, service_name: str) -> List[ServiceInfo]:
        """Discover instances of a service."""
        try:
            response = await self.client.get(
                f"{self.consul_url}/v1/health/service/{service_name}",
                params={"passing": "true"}  # Only healthy instances
            )
            
            if response.status_code == 200:
                services = response.json()
                service_instances = []
                
                for service in services:
                    service_data = service["Service"]
                    service_instances.append(ServiceInfo(
                        name=service_data["Service"],
                        id=service_data["ID"],
                        address=service_data["Address"],
                        port=service_data["Port"],
                        tags=service_data.get("Tags", [])
                    ))
                
                logger.info(f"Discovered {len(service_instances)} instances of {service_name}")
                return service_instances
            else:
                logger.error(f"Failed to discover service {service_name}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error discovering service {service_name}: {e}")
            return []
    
    async def get_service_url(self, service_name: str) -> Optional[str]:
        """Get URL for a service (returns first healthy instance)."""
        instances = await self.discover_service(service_name)
        
        if instances:
            instance = instances[0]  # Use first available instance
            return f"http://{instance.address}:{instance.port}"
        
        return None
    
    async def health_check(self) -> bool:
        """Check if Consul is available."""
        try:
            response = await self.client.get(f"{self.consul_url}/v1/status/leader")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Consul health check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Mock implementation for development when Consul is not available
class MockServiceDiscovery:
    """Mock service discovery for development."""
    
    def __init__(self, consul_url: str = "http://localhost:8500"):
        self.services: Dict[str, List[ServiceInfo]] = {}
        logger.info("Using mock service discovery (Consul not available)")
    
    async def register_service(
        self,
        name: str,
        port: int,
        address: str = "localhost",
        health_check_url: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Mock service registration."""
        service_id = f"{name}-{address}-{port}"
        service_info = ServiceInfo(
            name=name,
            id=service_id,
            address=address,
            port=port,
            tags=tags or [],
            health_check_url=health_check_url
        )
        
        if name not in self.services:
            self.services[name] = []
        
        self.services[name].append(service_info)
        logger.info(f"Mock registered service {name} with ID {service_id}")
        return True
    
    async def deregister_service(self, service_id: str) -> bool:
        """Mock service deregistration."""
        for service_name, instances in self.services.items():
            self.services[service_name] = [
                instance for instance in instances 
                if instance.id != service_id
            ]
        
        logger.info(f"Mock deregistered service {service_id}")
        return True
    
    async def discover_service(self, service_name: str) -> List[ServiceInfo]:
        """Mock service discovery."""
        return self.services.get(service_name, [])
    
    async def get_service_url(self, service_name: str) -> Optional[str]:
        """Mock get service URL."""
        instances = await self.discover_service(service_name)
        
        if instances:
            instance = instances[0]
            return f"http://{instance.address}:{instance.port}"
        
        # Return default URLs for known services
        service_defaults = {
            "user-service": "http://localhost:8001",
            "roleplay-service": "http://localhost:8002",
            "ai-service": "http://localhost:8003",
            "notification-service": "http://localhost:8004",
            "analytics-service": "http://localhost:8005"
        }
        
        return service_defaults.get(service_name)
    
    async def health_check(self) -> bool:
        """Mock health check."""
        return True
    
    async def close(self):
        """Mock close."""
        pass