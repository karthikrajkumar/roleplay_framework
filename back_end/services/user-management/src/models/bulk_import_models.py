"""
SQLAlchemy models for bulk user import operations.

This module contains database models for tracking bulk import jobs,
progress, errors, and user import batches with optimized performance.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Boolean, DateTime, Integer, Float, Text, JSON, 
    ForeignKey, Index, CheckConstraint, Enum, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid import uuid4
import uuid
import enum

from ..config.database import Base


class ImportStatus(enum.Enum):
    """Enumeration for import job statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATING = "validating"
    IMPORTING = "importing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL_SUCCESS = "partial_success"


class ValidationSeverity(enum.Enum):
    """Enumeration for validation error severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class BulkImportJobModel(Base):
    """
    SQLAlchemy model for bulk import jobs.
    
    Tracks the overall status and metadata of bulk user import operations.
    """
    __tablename__ = "bulk_import_jobs"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4,
        index=True
    )
    
    # Job metadata
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), index=True)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True), 
        nullable=True, 
        index=True
    )
    
    # File information
    file_name: Mapped[str] = mapped_column(String(255))
    file_size: Mapped[int] = mapped_column(BigInteger)  # bytes
    file_type: Mapped[str] = mapped_column(String(20))  # csv, xlsx, xls
    file_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 hash
    file_path: Mapped[str] = mapped_column(String(500))  # storage path
    
    # Status and progress
    status: Mapped[ImportStatus] = mapped_column(
        Enum(ImportStatus),
        default=ImportStatus.PENDING,
        index=True
    )
    progress_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str] = mapped_column(String(50), default="initialized")
    
    # Counts and statistics
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    processed_rows: Mapped[int] = mapped_column(Integer, default=0)
    valid_rows: Mapped[int] = mapped_column(Integer, default=0)
    invalid_rows: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_rows: Mapped[int] = mapped_column(Integer, default=0)
    imported_users: Mapped[int] = mapped_column(Integer, default=0)
    failed_imports: Mapped[int] = mapped_column(Integer, default=0)
    skipped_rows: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    estimated_completion_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # rows per second
    
    # Configuration
    import_options: Mapped[dict] = mapped_column(JSONB, default=dict)
    validation_rules: Mapped[dict] = mapped_column(JSONB, default=dict)
    field_mapping: Mapped[dict] = mapped_column(JSONB, default=dict)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    can_retry: Mapped[bool] = mapped_column(Boolean, default=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    
    # Rollback information
    rollback_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    can_rollback: Mapped[bool] = mapped_column(Boolean, default=False)
    rollback_deadline: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow,
        index=True
    )
    
    # Relationships
    batches: Mapped[List["ImportBatchModel"]] = relationship(
        "ImportBatchModel",
        back_populates="import_job",
        cascade="all, delete-orphan"
    )
    validation_errors: Mapped[List["ValidationErrorModel"]] = relationship(
        "ValidationErrorModel",
        back_populates="import_job",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_bulk_import_jobs_status_created", status, created_at.desc()),
        Index("idx_bulk_import_jobs_creator_org", created_by, organization_id),
        Index("idx_bulk_import_jobs_progress", progress_percentage, status),
        CheckConstraint(
            "file_type IN ('csv', 'xlsx', 'xls')",
            name="ck_bulk_import_jobs_file_type"
        ),
        CheckConstraint(
            "progress_percentage >= 0 AND progress_percentage <= 100",
            name="ck_bulk_import_jobs_progress"
        )
    )


class ImportBatchModel(Base):
    """
    SQLAlchemy model for import batches.
    
    Represents chunks of data being processed in parallel for efficient
    bulk import operations.
    """
    __tablename__ = "import_batches"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4,
        index=True
    )
    
    # Foreign key to import job
    import_job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("bulk_import_jobs.id", ondelete="CASCADE"),
        index=True
    )
    
    # Batch metadata
    batch_number: Mapped[int] = mapped_column(Integer, index=True)
    start_row: Mapped[int] = mapped_column(Integer)
    end_row: Mapped[int] = mapped_column(Integer)
    row_count: Mapped[int] = mapped_column(Integer)
    
    # Processing status
    status: Mapped[ImportStatus] = mapped_column(
        Enum(ImportStatus),
        default=ImportStatus.PENDING,
        index=True
    )
    worker_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    
    # Progress
    processed_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Data
    batch_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_user_ids: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String), 
        nullable=True
    )
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    
    # Relationships
    import_job: Mapped["BulkImportJobModel"] = relationship(
        "BulkImportJobModel",
        back_populates="batches"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_import_batches_job_batch", import_job_id, batch_number),
        Index("idx_import_batches_status_worker", status, worker_id),
        Index("idx_import_batches_row_range", start_row, end_row),
    )


class ValidationErrorModel(Base):
    """
    SQLAlchemy model for validation errors.
    
    Stores detailed validation errors encountered during bulk import.
    """
    __tablename__ = "validation_errors"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Foreign key to import job
    import_job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("bulk_import_jobs.id", ondelete="CASCADE"),
        index=True
    )
    
    # Error location
    row_number: Mapped[int] = mapped_column(Integer, index=True)
    column_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    field_path: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    
    # Error details
    error_code: Mapped[str] = mapped_column(String(50), index=True)
    error_message: Mapped[str] = mapped_column(Text)
    severity: Mapped[ValidationSeverity] = mapped_column(
        Enum(ValidationSeverity),
        default=ValidationSeverity.ERROR,
        index=True
    )
    
    # Context
    raw_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected_format: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    suggestion: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Additional data
    error_context: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    is_blocking: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    import_job: Mapped["BulkImportJobModel"] = relationship(
        "BulkImportJobModel",
        back_populates="validation_errors"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_validation_errors_job_row", import_job_id, row_number),
        Index("idx_validation_errors_severity_blocking", severity, is_blocking),
        Index("idx_validation_errors_code", error_code),
    )


class DuplicateDetectionCacheModel(Base):
    """
    SQLAlchemy model for duplicate detection cache.
    
    Stores hash-based signatures for efficient duplicate detection
    across large datasets.
    """
    __tablename__ = "duplicate_detection_cache"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Cache key (hash of normalized user data)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    
    # User identification
    email_hash: Mapped[str] = mapped_column(String(64), index=True)
    username_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    phone_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    
    # Source information
    source_type: Mapped[str] = mapped_column(String(20))  # 'database' or 'import'
    source_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    import_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("bulk_import_jobs.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    
    # Metadata
    user_data_signature: Mapped[str] = mapped_column(String(128))
    similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_duplicate_cache_email_source", email_hash, source_type),
        Index("idx_duplicate_cache_expires", expires_at),
        Index("idx_duplicate_cache_import_job", import_job_id),
        CheckConstraint(
            "source_type IN ('database', 'import')",
            name="ck_duplicate_cache_source_type"
        )
    )


class ImportProgressModel(Base):
    """
    SQLAlchemy model for real-time import progress tracking.
    
    Stores granular progress information for real-time updates
    and accurate time estimation.
    """
    __tablename__ = "import_progress"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Foreign key to import job
    import_job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("bulk_import_jobs.id", ondelete="CASCADE"),
        index=True
    )
    
    # Progress tracking
    stage: Mapped[str] = mapped_column(String(50), index=True)
    stage_progress: Mapped[float] = mapped_column(Float, default=0.0)
    overall_progress: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Performance metrics
    current_rate: Mapped[float] = mapped_column(Float, default=0.0)  # items per second
    average_rate: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_time_remaining: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # seconds
    
    # Current operation
    current_operation: Mapped[str] = mapped_column(String(200))
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_total: Mapped[int] = mapped_column(Integer, default=0)
    
    # Worker information
    worker_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    worker_count: Mapped[int] = mapped_column(Integer, default=1)
    
    # Additional metrics
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cpu_usage_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow,
        index=True
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_import_progress_job_updated", import_job_id, updated_at.desc()),
        Index("idx_import_progress_stage", stage),
        CheckConstraint(
            "stage_progress >= 0 AND stage_progress <= 100",
            name="ck_import_progress_stage"
        ),
        CheckConstraint(
            "overall_progress >= 0 AND overall_progress <= 100",
            name="ck_import_progress_overall"
        )
    )