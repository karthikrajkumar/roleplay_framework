from typing import Dict, Any


class Container:
    """Simple dependency injection container for the user management service."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
            cls._instance._singletons = {}
        return cls._instance
    
    def register(self, name: str, service: Any, singleton: bool = False):
        """Register a service in the container."""
        if singleton:
            self._singletons[name] = service
        else:
            self._services[name] = service
            
    @classmethod
    def register_singleton(cls, interface, implementation):
        """Register a singleton service by interface."""
        instance = cls()
        key = interface.__name__ if hasattr(interface, '__name__') else str(interface)
        instance._singletons[key] = implementation
    
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