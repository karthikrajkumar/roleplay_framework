"""
Storage interface definitions for file and vector storage.

This module defines interfaces for file storage and vector database
operations to support multimedia content and AI embeddings.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, BinaryIO, Union
from uuid import UUID
from datetime import datetime
from enum import Enum


class StorageType(str, Enum):
    """Storage type enumeration."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class FileMetadata:
    """File metadata container."""
    
    def __init__(
        self,
        filename: str,
        size: int,
        content_type: str,
        upload_date: datetime,
        checksum: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        self.filename = filename
        self.size = size
        self.content_type = content_type
        self.upload_date = upload_date
        self.checksum = checksum
        self.tags = tags or {}


class IFileStorage(ABC):
    """
    Interface for file storage operations.
    
    Provides abstraction for different file storage backends
    including local storage, S3, Google Cloud Storage, etc.
    """
    
    @abstractmethod
    async def upload_file(
        self,
        file_path: str,
        content: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload file and return file URL/path."""
        pass
    
    @abstractmethod
    async def download_file(self, file_path: str) -> bytes:
        """Download file content."""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file."""
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Get file metadata."""
        pass
    
    @abstractmethod
    async def list_files(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[str]:
        """List files with optional prefix filter."""
        pass
    
    @abstractmethod
    async def get_file_url(
        self,
        file_path: str,
        expires_in_seconds: Optional[int] = None
    ) -> str:
        """Get signed URL for file access."""
        pass
    
    @abstractmethod
    async def copy_file(self, source_path: str, destination_path: str) -> bool:
        """Copy file to new location."""
        pass
    
    @abstractmethod
    async def move_file(self, source_path: str, destination_path: str) -> bool:
        """Move file to new location."""
        pass
    
    @abstractmethod
    async def get_file_size(self, file_path: str) -> Optional[int]:
        """Get file size in bytes."""
        pass


class IAvatarStorage(IFileStorage):
    """
    Specialized interface for avatar/image storage.
    
    Extends file storage with image-specific operations
    like resizing, format conversion, and optimization.
    """
    
    @abstractmethod
    async def upload_avatar(
        self,
        user_id: UUID,
        image_content: bytes,
        original_filename: str
    ) -> str:
        """Upload and process user avatar."""
        pass
    
    @abstractmethod
    async def generate_thumbnail(
        self,
        image_path: str,
        width: int = 150,
        height: int = 150
    ) -> str:
        """Generate thumbnail for image."""
        pass
    
    @abstractmethod
    async def resize_image(
        self,
        image_path: str,
        width: int,
        height: int,
        maintain_aspect_ratio: bool = True
    ) -> str:
        """Resize image to specified dimensions."""
        pass
    
    @abstractmethod
    async def convert_format(
        self,
        image_path: str,
        target_format: str
    ) -> str:
        """Convert image to different format."""
        pass
    
    @abstractmethod
    async def optimize_image(
        self,
        image_path: str,
        quality: int = 85
    ) -> str:
        """Optimize image for web delivery."""
        pass


class VectorSearchResult:
    """Vector search result container."""
    
    def __init__(
        self,
        id: str,
        score: float,
        metadata: Dict[str, Any],
        content: Optional[str] = None
    ):
        self.id = id
        self.score = score
        self.metadata = metadata
        self.content = content


class IVectorStore(ABC):
    """
    Interface for vector database operations.
    
    Provides abstraction for vector databases like Pinecone,
    Weaviate, or Chroma for similarity search and embeddings.
    """
    
    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    async def update_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update existing vector."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Get vector by ID."""
        pass
    
    @abstractmethod
    async def get_vectors(self, ids: List[str]) -> Dict[str, List[float]]:
        """Get multiple vectors by IDs."""
        pass
    
    @abstractmethod
    async def count_vectors(
        self,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count vectors with optional filter."""
        pass
    
    @abstractmethod
    async def list_vectors(
        self,
        limit: int = 100,
        offset: int = 0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """List vector IDs with pagination."""
        pass


class IEmbeddingService(ABC):
    """
    Interface for text embedding generation.
    
    Provides abstraction for different embedding models
    and services for generating vector representations.
    """
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Get embedding vector dimension."""
        pass
    
    @abstractmethod
    async def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate similarity between two embeddings."""
        pass


class ICharacterEmbeddingService(IEmbeddingService):
    """
    Specialized interface for character embedding operations.
    
    Handles character-specific embedding generation for
    similarity matching and recommendation systems.
    """
    
    @abstractmethod
    async def embed_character(self, character_id: UUID) -> List[float]:
        """Generate embedding for character profile."""
        pass
    
    @abstractmethod
    async def find_similar_characters(
        self,
        character_id: UUID,
        top_k: int = 10,
        exclude_creator: bool = False
    ) -> List[VectorSearchResult]:
        """Find characters similar to given character."""
        pass
    
    @abstractmethod
    async def recommend_characters_for_scenario(
        self,
        scenario_id: UUID,
        top_k: int = 5
    ) -> List[VectorSearchResult]:
        """Recommend characters for a scenario."""
        pass
    
    @abstractmethod
    async def update_character_embedding(self, character_id: UUID) -> bool:
        """Update character embedding after profile changes."""
        pass


class IScenarioEmbeddingService(IEmbeddingService):
    """
    Specialized interface for scenario embedding operations.
    
    Handles scenario-specific embedding generation for
    content discovery and matching systems.
    """
    
    @abstractmethod
    async def embed_scenario(self, scenario_id: UUID) -> List[float]:
        """Generate embedding for scenario."""
        pass
    
    @abstractmethod
    async def find_similar_scenarios(
        self,
        scenario_id: UUID,
        top_k: int = 10
    ) -> List[VectorSearchResult]:
        """Find scenarios similar to given scenario."""
        pass
    
    @abstractmethod
    async def recommend_scenarios_for_user(
        self,
        user_id: UUID,
        top_k: int = 10
    ) -> List[VectorSearchResult]:
        """Recommend scenarios based on user preferences."""
        pass
    
    @abstractmethod
    async def search_scenarios_by_description(
        self,
        description: str,
        top_k: int = 10
    ) -> List[VectorSearchResult]:
        """Search scenarios by natural language description."""
        pass