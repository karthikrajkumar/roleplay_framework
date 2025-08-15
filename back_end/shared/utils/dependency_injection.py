"""
Dependency injection container and decorators.

This module provides a lightweight dependency injection system
for managing service dependencies and promoting loose coupling.
"""

import inspect
from typing import Any, Callable, Dict, Type, TypeVar, Union, get_type_hints
from functools import wraps
from abc import ABC, abstractmethod


T = TypeVar('T')


class DIContainer:
    """
    Dependency injection container for managing service instances.
    
    Supports singleton and transient lifetimes, factory functions,
    and interface-to-implementation mapping.
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._interfaces: Dict[Type, Type] = {}
    
    def register_singleton(self, interface: Type[T], implementation: Union[T, Type[T]]) -> None:
        """Register a singleton service."""
        key = self._get_key(interface)
        if inspect.isclass(implementation):
            self._services[key] = implementation
        else:
            self._singletons[key] = implementation
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service (new instance each time)."""
        key = self._get_key(interface)
        self._services[key] = implementation
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for creating instances."""
        key = self._get_key(interface)
        self._factories[key] = factory
    
    def register_interface(self, interface: Type, implementation: Type) -> None:
        """Register an interface-to-implementation mapping."""
        self._interfaces[interface] = implementation
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance."""
        key = self._get_key(interface)
        
        # Check for existing singleton
        if key in self._singletons:
            return self._singletons[key]
        
        # Check for factory
        if key in self._factories:
            instance = self._factories[key]()
            return instance
        
        # Check for registered service
        if key in self._services:
            service_class = self._services[key]
            
            # Create instance with dependency injection
            instance = self._create_instance(service_class)
            
            # Store as singleton if registered as such
            if key in self._services:
                self._singletons[key] = instance
            
            return instance
        
        # Check for interface mapping
        if interface in self._interfaces:
            implementation = self._interfaces[interface]
            return self.resolve(implementation)
        
        # Try to create instance directly
        try:
            return self._create_instance(interface)
        except Exception as e:
            raise ValueError(f"Cannot resolve service {interface}: {e}")
    
    def _create_instance(self, service_class: Type[T]) -> T:
        """Create instance with dependency injection."""
        try:
            # Get constructor signature
            sig = inspect.signature(service_class.__init__)
            params = {}
            
            # Resolve dependencies
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation == inspect.Parameter.empty:
                    continue
                
                # Resolve dependency
                dependency = self.resolve(param.annotation)
                params[param_name] = dependency
            
            return service_class(**params)
        except Exception as e:
            raise ValueError(f"Failed to create instance of {service_class}: {e}")
    
    def _get_key(self, interface: Type) -> str:
        """Get string key for interface."""
        return f"{interface.__module__}.{interface.__qualname__}"
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()
        self._interfaces.clear()


# Global container instance
Container = DIContainer()


def inject(func: Callable) -> Callable:
    """
    Decorator for automatic dependency injection.
    
    Automatically resolves and injects dependencies based on type hints.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Get type hints for function parameters
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        
        # Resolve dependencies that aren't provided
        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                continue  # Already provided
            
            if param_name in type_hints:
                dependency_type = type_hints[param_name]
                try:
                    dependency = Container.resolve(dependency_type)
                    kwargs[param_name] = dependency
                except ValueError:
                    # Dependency not registered, skip
                    pass
        
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Get type hints for function parameters
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        
        # Resolve dependencies that aren't provided
        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                continue  # Already provided
            
            if param_name in type_hints:
                dependency_type = type_hints[param_name]
                try:
                    dependency = Container.resolve(dependency_type)
                    kwargs[param_name] = dependency
                except ValueError:
                    # Dependency not registered, skip
                    pass
        
        return func(*args, **kwargs)
    
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class ServiceRegistry:
    """
    Service registry for automatic service discovery and registration.
    
    Provides decorators and utilities for registering services
    with the dependency injection container.
    """
    
    @staticmethod
    def singleton(interface: Type = None):
        """Decorator to register a class as a singleton service."""
        def decorator(cls):
            target_interface = interface or cls
            Container.register_singleton(target_interface, cls)
            return cls
        return decorator
    
    @staticmethod
    def transient(interface: Type = None):
        """Decorator to register a class as a transient service."""
        def decorator(cls):
            target_interface = interface or cls
            Container.register_transient(target_interface, cls)
            return cls
        return decorator
    
    @staticmethod
    def factory(interface: Type, factory_func: Callable):
        """Register a factory function for a service."""
        Container.register_factory(interface, factory_func)
    
    @staticmethod
    def bind(interface: Type, implementation: Type):
        """Bind an interface to its implementation."""
        Container.register_interface(interface, implementation)


# Convenience aliases
singleton = ServiceRegistry.singleton
transient = ServiceRegistry.transient
factory = ServiceRegistry.factory
bind = ServiceRegistry.bind