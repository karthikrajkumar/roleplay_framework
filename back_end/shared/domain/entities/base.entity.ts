/**
 * Base Entity - Abstract base class for all domain entities
 * Implements common entity behaviors and ensures consistent ID handling
 */

export abstract class BaseEntity {
  protected _id: string;
  protected _createdAt: Date;
  protected _updatedAt: Date;
  protected _version: number;

  constructor(id: string) {
    this._id = id;
    this._createdAt = new Date();
    this._updatedAt = new Date();
    this._version = 1;
  }

  get id(): string {
    return this._id;
  }

  get createdAt(): Date {
    return this._createdAt;
  }

  get updatedAt(): Date {
    return this._updatedAt;
  }

  get version(): number {
    return this._version;
  }

  protected markModified(): void {
    this._updatedAt = new Date();
    this._version++;
  }

  abstract validate(): boolean;
  abstract toDTO(): Record<string, any>;
}