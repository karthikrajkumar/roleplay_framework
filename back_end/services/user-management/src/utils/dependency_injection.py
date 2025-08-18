from typing import Dict, Any


class Container:
    """Simple dependency injection container for the user management service."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, name: str, service: Any, singleton: bool = False):
        """Register a service in the container."""
        if singleton:
            self._singletons[name] = service
        else:
            self._services[name] = service
    
    def get(self, name: str) -> Any:
        """Get a service from the container."""
        if name in self._singletons:
            return self._singletons[name]
        elif name in self._services:
            return self._services[name]
        else:
            raise KeyError(f"Service '{name}' not found in container")
    
    def has(self, name: str) -> bool:
        """Check if a service is registered in the container."""
        return name in self._services or name in self._singletons


# Global container instance
container = Container()