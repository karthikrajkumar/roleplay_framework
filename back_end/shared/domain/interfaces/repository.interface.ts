/**
 * Repository Pattern Interfaces
 * Defines contracts for data access across all services
 */

export interface IBaseRepository<TEntity, TId> {
  findById(id: TId): Promise<TEntity | null>;
  findAll(options?: QueryOptions): Promise<TEntity[]>;
  create(entity: TEntity): Promise<TEntity>;
  update(id: TId, entity: Partial<TEntity>): Promise<TEntity>;
  delete(id: TId): Promise<boolean>;
  exists(id: TId): Promise<boolean>;
  count(filter?: FilterOptions): Promise<number>;
}

export interface IReadOnlyRepository<TEntity, TId> {
  findById(id: TId): Promise<TEntity | null>;
  findAll(options?: QueryOptions): Promise<TEntity[]>;
  exists(id: TId): Promise<boolean>;
  count(filter?: FilterOptions): Promise<number>;
}

export interface QueryOptions {
  limit?: number;
  offset?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  filters?: FilterOptions;
}

export interface FilterOptions {
  [key: string]: any;
}

export interface IUnitOfWork {
  begin(): Promise<void>;
  commit(): Promise<void>;
  rollback(): Promise<void>;
  isActive(): boolean;
}