-- =============================================================================
-- User Management System Schema Extensions
-- Multi-tenant Organizations, Groups, SSO, and Bulk Import Support
-- PostgreSQL Database Schema with UUID, JSONB, and Performance Optimizations
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For text search optimization
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For composite indexes with JSONB

-- =============================================================================
-- 1. ORGANIZATIONS TABLE - Multi-tenant Support
-- =============================================================================

CREATE TABLE organizations (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL, -- URL-friendly identifier
    display_name VARCHAR(255),
    
    -- Organization metadata
    description TEXT,
    website VARCHAR(255),
    industry VARCHAR(100),
    company_size VARCHAR(50), -- 'startup', 'small', 'medium', 'large', 'enterprise'
    
    -- Contact information
    primary_email VARCHAR(255),
    primary_phone VARCHAR(20),
    primary_contact_name VARCHAR(255),
    
    -- Address (JSONB for flexibility across regions)
    address_data JSONB DEFAULT '{}', -- {street, city, state, postal_code, country}
    
    -- Billing and subscription
    billing_email VARCHAR(255),
    subscription_tier VARCHAR(50) DEFAULT 'basic' NOT NULL,
    subscription_status VARCHAR(30) DEFAULT 'active' NOT NULL,
    subscription_expires_at TIMESTAMP WITH TIME ZONE,
    trial_ends_at TIMESTAMP WITH TIME ZONE,
    max_users INTEGER DEFAULT 10,
    max_groups INTEGER DEFAULT 5,
    
    -- Organization settings (JSONB for extensibility)
    settings JSONB DEFAULT '{}', -- branding, features, integrations, etc.
    feature_flags JSONB DEFAULT '{}', -- enabled features
    
    -- Security and compliance
    sso_enabled BOOLEAN DEFAULT FALSE,
    sso_config JSONB DEFAULT '{}', -- SSO configuration
    security_settings JSONB DEFAULT '{}', -- password policies, MFA requirements, etc.
    compliance_settings JSONB DEFAULT '{}', -- GDPR, HIPAA, etc.
    
    -- Usage tracking
    total_users INTEGER DEFAULT 0,
    total_groups INTEGER DEFAULT 0,
    total_invitations_sent INTEGER DEFAULT 0,
    monthly_active_users INTEGER DEFAULT 0,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    updated_by UUID,
    version INTEGER DEFAULT 1,
    status VARCHAR(30) DEFAULT 'active' NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_organizations_subscription_tier 
        CHECK (subscription_tier IN ('trial', 'basic', 'premium', 'enterprise')),
    CONSTRAINT ck_organizations_subscription_status 
        CHECK (subscription_status IN ('active', 'expired', 'cancelled', 'suspended')),
    CONSTRAINT ck_organizations_company_size 
        CHECK (company_size IN ('startup', 'small', 'medium', 'large', 'enterprise')),
    CONSTRAINT ck_organizations_status 
        CHECK (status IN ('active', 'inactive', 'suspended', 'deleted')),
    CONSTRAINT ck_organizations_max_users_positive 
        CHECK (max_users > 0),
    CONSTRAINT ck_organizations_max_groups_positive 
        CHECK (max_groups > 0)
);

-- Performance indexes for organizations
CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_name_trgm ON organizations USING gin(name gin_trgm_ops);
CREATE INDEX idx_organizations_subscription ON organizations(subscription_tier, subscription_status);
CREATE INDEX idx_organizations_expires_at ON organizations(subscription_expires_at) WHERE subscription_expires_at IS NOT NULL;
CREATE INDEX idx_organizations_active ON organizations(status) WHERE status = 'active';
CREATE INDEX idx_organizations_settings ON organizations USING gin(settings);
CREATE INDEX idx_organizations_created_at ON organizations(created_at DESC);

-- =============================================================================
-- 2. ORGANIZATION MEMBERSHIPS - User-Organization Relationships
-- =============================================================================

CREATE TABLE organization_memberships (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Membership details
    role VARCHAR(50) DEFAULT 'member' NOT NULL,
    title VARCHAR(100), -- Job title within organization
    department VARCHAR(100),
    
    -- Status and permissions
    status VARCHAR(30) DEFAULT 'active' NOT NULL,
    permissions JSONB DEFAULT '{}', -- Role-specific permissions
    access_level INTEGER DEFAULT 1, -- Numeric access level (1-10)
    
    -- Membership timeline
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    left_at TIMESTAMP WITH TIME ZONE,
    last_activity TIMESTAMP WITH TIME ZONE,
    
    -- Invitation tracking
    invited_by UUID REFERENCES users(id),
    invitation_accepted_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    updated_by UUID,
    
    -- Constraints
    UNIQUE(organization_id, user_id),
    CONSTRAINT ck_org_memberships_role 
        CHECK (role IN ('owner', 'admin', 'manager', 'member', 'guest', 'billing_admin')),
    CONSTRAINT ck_org_memberships_status 
        CHECK (status IN ('active', 'inactive', 'pending', 'suspended', 'left')),
    CONSTRAINT ck_org_memberships_access_level 
        CHECK (access_level BETWEEN 1 AND 10),
    CONSTRAINT ck_org_memberships_left_after_joined 
        CHECK (left_at IS NULL OR left_at >= joined_at)
);

-- Performance indexes for organization memberships
CREATE INDEX idx_org_memberships_organization ON organization_memberships(organization_id, status);
CREATE INDEX idx_org_memberships_user ON organization_memberships(user_id, status);
CREATE INDEX idx_org_memberships_role ON organization_memberships(role, status);
CREATE INDEX idx_org_memberships_joined ON organization_memberships(joined_at DESC);
CREATE INDEX idx_org_memberships_permissions ON organization_memberships USING gin(permissions);

-- =============================================================================
-- 3. USER GROUPS - Content Creator Groups
-- =============================================================================

CREATE TABLE user_groups (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Group identification
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL, -- URL-friendly identifier within organization
    display_name VARCHAR(255),
    
    -- Group metadata
    description TEXT,
    category VARCHAR(100), -- 'course', 'team', 'project', 'department', etc.
    tags TEXT[], -- Searchable tags
    
    -- Group settings
    visibility VARCHAR(30) DEFAULT 'private' NOT NULL,
    join_policy VARCHAR(30) DEFAULT 'invitation' NOT NULL,
    max_members INTEGER DEFAULT 100,
    auto_archive_days INTEGER, -- Auto-archive after inactivity
    
    -- Content and features
    features_enabled JSONB DEFAULT '{}', -- Available features for this group
    group_settings JSONB DEFAULT '{}', -- Group-specific settings
    
    -- Usage tracking
    total_members INTEGER DEFAULT 0,
    active_members INTEGER DEFAULT 0,
    total_content_items INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE,
    
    -- Group lifecycle
    archived_at TIMESTAMP WITH TIME ZONE,
    archived_by UUID REFERENCES users(id),
    archive_reason TEXT,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID NOT NULL REFERENCES users(id), -- Content creator who owns the group
    updated_by UUID,
    version INTEGER DEFAULT 1,
    status VARCHAR(30) DEFAULT 'active' NOT NULL,
    
    -- Constraints
    UNIQUE(organization_id, slug),
    CONSTRAINT ck_user_groups_visibility 
        CHECK (visibility IN ('public', 'organization', 'private', 'secret')),
    CONSTRAINT ck_user_groups_join_policy 
        CHECK (join_policy IN ('open', 'request', 'invitation', 'closed')),
    CONSTRAINT ck_user_groups_status 
        CHECK (status IN ('active', 'archived', 'suspended', 'deleted')),
    CONSTRAINT ck_user_groups_max_members_positive 
        CHECK (max_members > 0)
);

-- Performance indexes for user groups
CREATE INDEX idx_user_groups_organization ON user_groups(organization_id, status);
CREATE INDEX idx_user_groups_slug ON user_groups(organization_id, slug);
CREATE INDEX idx_user_groups_creator ON user_groups(created_by, status);
CREATE INDEX idx_user_groups_category ON user_groups(category, visibility);
CREATE INDEX idx_user_groups_tags ON user_groups USING gin(tags);
CREATE INDEX idx_user_groups_name_trgm ON user_groups USING gin(name gin_trgm_ops);
CREATE INDEX idx_user_groups_activity ON user_groups(last_activity DESC) WHERE status = 'active';
CREATE INDEX idx_user_groups_settings ON user_groups USING gin(group_settings);

-- =============================================================================
-- 4. GROUP MEMBERSHIPS - User-Group Relationships with Roles
-- =============================================================================

CREATE TABLE group_memberships (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    group_id UUID NOT NULL REFERENCES user_groups(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Membership details
    role VARCHAR(50) DEFAULT 'learner' NOT NULL,
    access_level INTEGER DEFAULT 1, -- Numeric access level for permissions
    
    -- Status and progress
    status VARCHAR(30) DEFAULT 'active' NOT NULL,
    progress_data JSONB DEFAULT '{}', -- Learning progress, completion status, etc.
    participation_score DECIMAL(5,2) DEFAULT 0.0, -- 0-100 participation score
    
    -- Membership timeline
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    left_at TIMESTAMP WITH TIME ZONE,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Invitation tracking
    invited_by UUID REFERENCES users(id),
    invitation_accepted_at TIMESTAMP WITH TIME ZONE,
    
    -- Activity tracking
    total_contributions INTEGER DEFAULT 0,
    total_content_consumed INTEGER DEFAULT 0,
    streak_days INTEGER DEFAULT 0,
    last_contribution_at TIMESTAMP WITH TIME ZONE,
    
    -- Custom metadata
    member_metadata JSONB DEFAULT '{}', -- Role-specific or custom data
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    updated_by UUID,
    
    -- Constraints
    UNIQUE(group_id, user_id),
    CONSTRAINT ck_group_memberships_role 
        CHECK (role IN ('owner', 'admin', 'creator', 'moderator', 'learner', 'observer')),
    CONSTRAINT ck_group_memberships_status 
        CHECK (status IN ('active', 'inactive', 'pending', 'suspended', 'left', 'banned')),
    CONSTRAINT ck_group_memberships_access_level 
        CHECK (access_level BETWEEN 1 AND 10),
    CONSTRAINT ck_group_memberships_participation_score 
        CHECK (participation_score BETWEEN 0.0 AND 100.0),
    CONSTRAINT ck_group_memberships_left_after_joined 
        CHECK (left_at IS NULL OR left_at >= joined_at)
);

-- Performance indexes for group memberships
CREATE INDEX idx_group_memberships_group ON group_memberships(group_id, status);
CREATE INDEX idx_group_memberships_user ON group_memberships(user_id, status);
CREATE INDEX idx_group_memberships_role ON group_memberships(role, status);
CREATE INDEX idx_group_memberships_activity ON group_memberships(last_activity DESC) WHERE status = 'active';
CREATE INDEX idx_group_memberships_progress ON group_memberships USING gin(progress_data);
CREATE INDEX idx_group_memberships_participation ON group_memberships(participation_score DESC) WHERE status = 'active';

-- =============================================================================
-- 5. BULK IMPORT OPERATIONS - Track Import Jobs
-- =============================================================================

CREATE TABLE import_operations (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Import metadata
    operation_type VARCHAR(50) NOT NULL, -- 'users', 'groups', 'memberships', etc.
    file_name VARCHAR(255),
    file_size BIGINT,
    file_hash VARCHAR(64), -- SHA-256 hash for deduplication
    
    -- Import configuration
    import_config JSONB DEFAULT '{}', -- Column mappings, validation rules, etc.
    field_mappings JSONB DEFAULT '{}', -- CSV column to database field mappings
    
    -- Status and progress
    status VARCHAR(30) DEFAULT 'pending' NOT NULL,
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    total_records INTEGER DEFAULT 0,
    processed_records INTEGER DEFAULT 0,
    successful_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,
    
    -- Processing details
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion_at TIMESTAMP WITH TIME ZONE,
    processing_time_seconds INTEGER,
    
    -- Error tracking
    error_summary JSONB DEFAULT '{}', -- Summary of error types and counts
    error_details TEXT[], -- Detailed error messages
    validation_errors JSONB DEFAULT '{}', -- Field-level validation errors
    
    -- Results and output
    results_summary JSONB DEFAULT '{}', -- Summary of import results
    output_file_path VARCHAR(500), -- Path to detailed results file
    rollback_data JSONB, -- Data needed for rollback if supported
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID NOT NULL REFERENCES users(id),
    updated_by UUID,
    
    -- Constraints
    CONSTRAINT ck_import_operations_type 
        CHECK (operation_type IN ('users', 'groups', 'memberships', 'invitations', 'mixed')),
    CONSTRAINT ck_import_operations_status 
        CHECK (status IN ('pending', 'validating', 'processing', 'completed', 'failed', 'cancelled', 'rollback')),
    CONSTRAINT ck_import_operations_progress 
        CHECK (progress_percentage BETWEEN 0.0 AND 100.0),
    CONSTRAINT ck_import_operations_completed_after_started 
        CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
);

-- Performance indexes for import operations
CREATE INDEX idx_import_operations_organization ON import_operations(organization_id, status);
CREATE INDEX idx_import_operations_status ON import_operations(status, created_at DESC);
CREATE INDEX idx_import_operations_type ON import_operations(operation_type, status);
CREATE INDEX idx_import_operations_creator ON import_operations(created_by, created_at DESC);
CREATE INDEX idx_import_operations_file_hash ON import_operations(file_hash) WHERE file_hash IS NOT NULL;
CREATE INDEX idx_import_operations_progress ON import_operations(progress_percentage) WHERE status IN ('processing', 'validating');

-- =============================================================================
-- 6. EXTERNAL IDENTITIES - SSO Identity Mapping
-- =============================================================================

CREATE TABLE external_identities (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- External identity details
    provider VARCHAR(100) NOT NULL, -- 'google', 'microsoft', 'okta', 'saml', etc.
    provider_user_id VARCHAR(255) NOT NULL, -- Unique ID from external provider
    provider_username VARCHAR(255),
    provider_email VARCHAR(255),
    provider_display_name VARCHAR(255),
    
    -- Identity metadata from provider
    provider_metadata JSONB DEFAULT '{}', -- Additional user data from provider
    profile_data JSONB DEFAULT '{}', -- Cached profile information
    
    -- Authentication details
    access_token_hash VARCHAR(255), -- Hashed access token
    refresh_token_hash VARCHAR(255), -- Hashed refresh token
    token_expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Linking and verification
    identity_verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMP WITH TIME ZONE,
    link_method VARCHAR(50), -- 'sso_signup', 'account_linking', 'admin_created'
    
    -- Status and usage
    status VARCHAR(30) DEFAULT 'active' NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    linked_by UUID REFERENCES users(id), -- User who linked this identity
    
    -- Constraints
    UNIQUE(provider, provider_user_id),
    UNIQUE(user_id, provider, organization_id), -- One identity per provider per org per user
    CONSTRAINT ck_external_identities_provider 
        CHECK (provider IN ('google', 'microsoft', 'github', 'okta', 'auth0', 'saml', 'ldap', 'custom')),
    CONSTRAINT ck_external_identities_status 
        CHECK (status IN ('active', 'inactive', 'suspended', 'revoked')),
    CONSTRAINT ck_external_identities_link_method 
        CHECK (link_method IN ('sso_signup', 'account_linking', 'admin_created', 'bulk_import'))
);

-- Performance indexes for external identities
CREATE INDEX idx_external_identities_user ON external_identities(user_id, status);
CREATE INDEX idx_external_identities_provider ON external_identities(provider, provider_user_id);
CREATE INDEX idx_external_identities_organization ON external_identities(organization_id, provider) WHERE organization_id IS NOT NULL;
CREATE INDEX idx_external_identities_email ON external_identities(provider_email) WHERE provider_email IS NOT NULL;
CREATE INDEX idx_external_identities_login ON external_identities(last_login DESC) WHERE status = 'active';
CREATE INDEX idx_external_identities_metadata ON external_identities USING gin(provider_metadata);

-- =============================================================================
-- 7. INVITATIONS - Pending User Invitations and Onboarding
-- =============================================================================

CREATE TABLE invitations (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Invitation target
    email VARCHAR(255) NOT NULL,
    invited_user_id UUID REFERENCES users(id), -- Set when user already exists
    
    -- Invitation details
    invitation_type VARCHAR(50) NOT NULL, -- 'organization', 'group', 'both'
    target_role VARCHAR(50), -- Role in organization
    
    -- Group invitation details (if applicable)
    target_group_id UUID REFERENCES user_groups(id) ON DELETE CASCADE,
    group_role VARCHAR(50), -- Role in group
    
    -- Invitation metadata
    personal_message TEXT,
    custom_data JSONB DEFAULT '{}', -- Custom invitation data
    
    -- Security and verification
    invitation_token VARCHAR(255) UNIQUE NOT NULL, -- Secure token for acceptance
    token_hash VARCHAR(255) UNIQUE NOT NULL, -- Hashed token for security
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days'),
    
    -- Status and tracking
    status VARCHAR(30) DEFAULT 'pending' NOT NULL,
    sent_count INTEGER DEFAULT 0,
    last_sent_at TIMESTAMP WITH TIME ZONE,
    opened_count INTEGER DEFAULT 0,
    first_opened_at TIMESTAMP WITH TIME ZONE,
    last_opened_at TIMESTAMP WITH TIME ZONE,
    
    -- Response tracking
    accepted_at TIMESTAMP WITH TIME ZONE,
    declined_at TIMESTAMP WITH TIME ZONE,
    declined_reason TEXT,
    
    -- Onboarding tracking
    onboarding_completed BOOLEAN DEFAULT FALSE,
    onboarding_completed_at TIMESTAMP WITH TIME ZONE,
    onboarding_progress JSONB DEFAULT '{}', -- Step-by-step progress
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    invited_by UUID NOT NULL REFERENCES users(id),
    updated_by UUID,
    
    -- Constraints
    CONSTRAINT ck_invitations_type 
        CHECK (invitation_type IN ('organization', 'group', 'both')),
    CONSTRAINT ck_invitations_status 
        CHECK (status IN ('pending', 'sent', 'opened', 'accepted', 'declined', 'expired', 'cancelled')),
    CONSTRAINT ck_invitations_target_role 
        CHECK (target_role IN ('owner', 'admin', 'manager', 'member', 'guest')),
    CONSTRAINT ck_invitations_group_role 
        CHECK (group_role IN ('owner', 'admin', 'creator', 'moderator', 'learner', 'observer')),
    CONSTRAINT ck_invitations_expires_future 
        CHECK (expires_at > created_at),
    CONSTRAINT ck_invitations_group_type_consistency 
        CHECK ((invitation_type IN ('group', 'both')) = (target_group_id IS NOT NULL))
);

-- Performance indexes for invitations
CREATE INDEX idx_invitations_email ON invitations(email, status);
CREATE INDEX idx_invitations_organization ON invitations(organization_id, status);
CREATE INDEX idx_invitations_token_hash ON invitations(token_hash);
CREATE INDEX idx_invitations_inviter ON invitations(invited_by, created_at DESC);
CREATE INDEX idx_invitations_expires ON invitations(expires_at) WHERE status IN ('pending', 'sent', 'opened');
CREATE INDEX idx_invitations_status ON invitations(status, created_at DESC);
CREATE INDEX idx_invitations_group ON invitations(target_group_id, status) WHERE target_group_id IS NOT NULL;
CREATE INDEX idx_invitations_user ON invitations(invited_user_id, status) WHERE invited_user_id IS NOT NULL;

-- =============================================================================
-- DATA PARTITIONING STRATEGIES FOR LARGE DATASETS
-- =============================================================================

-- Partition user_sessions by created_at (monthly partitions)
-- This is useful for session data that grows rapidly
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    -- Create partitioned table for new sessions (implement gradually)
    -- For now, we'll add comments for future partitioning strategy
    NULL;
END $$;

-- Add comments for future partitioning considerations
COMMENT ON TABLE user_sessions IS 'Consider partitioning by created_at monthly for large datasets';
COMMENT ON TABLE import_operations IS 'Consider partitioning by created_at monthly for high-volume imports';

-- =============================================================================
-- MATERIALIZED VIEWS FOR ANALYTICS AND PERFORMANCE
-- =============================================================================

-- Organization usage summary view
CREATE MATERIALIZED VIEW organization_usage_summary AS
SELECT 
    o.id as organization_id,
    o.name as organization_name,
    o.subscription_tier,
    COUNT(DISTINCT om.user_id) as total_users,
    COUNT(DISTINCT CASE WHEN om.status = 'active' THEN om.user_id END) as active_users,
    COUNT(DISTINCT ug.id) as total_groups,
    COUNT(DISTINCT CASE WHEN ug.status = 'active' THEN ug.id END) as active_groups,
    COUNT(DISTINCT gm.user_id) as total_group_members,
    COUNT(DISTINCT CASE WHEN gm.status = 'active' THEN gm.user_id END) as active_group_members,
    COUNT(DISTINCT i.id) as total_invitations,
    COUNT(DISTINCT CASE WHEN i.status = 'pending' THEN i.id END) as pending_invitations,
    o.updated_at as last_updated
FROM organizations o
LEFT JOIN organization_memberships om ON o.id = om.organization_id
LEFT JOIN user_groups ug ON o.id = ug.organization_id  
LEFT JOIN group_memberships gm ON ug.id = gm.group_id
LEFT JOIN invitations i ON o.id = i.organization_id
WHERE o.status = 'active'
GROUP BY o.id, o.name, o.subscription_tier, o.updated_at;

-- Index for the materialized view
CREATE INDEX idx_org_usage_summary_org_id ON organization_usage_summary(organization_id);

-- =============================================================================
-- TRIGGERS FOR AUTOMATED UPDATES
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to all tables
CREATE TRIGGER trigger_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_organization_memberships_updated_at BEFORE UPDATE ON organization_memberships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_user_groups_updated_at BEFORE UPDATE ON user_groups
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_group_memberships_updated_at BEFORE UPDATE ON group_memberships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_import_operations_updated_at BEFORE UPDATE ON import_operations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_external_identities_updated_at BEFORE UPDATE ON external_identities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_invitations_updated_at BEFORE UPDATE ON invitations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update organization user counts
CREATE OR REPLACE FUNCTION update_organization_user_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        UPDATE organizations 
        SET total_users = (
            SELECT COUNT(*) 
            FROM organization_memberships 
            WHERE organization_id = NEW.organization_id 
            AND status = 'active'
        )
        WHERE id = NEW.organization_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE organizations 
        SET total_users = (
            SELECT COUNT(*) 
            FROM organization_memberships 
            WHERE organization_id = OLD.organization_id 
            AND status = 'active'
        )
        WHERE id = OLD.organization_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger to maintain organization user counts
CREATE TRIGGER trigger_update_org_user_count
    AFTER INSERT OR UPDATE OR DELETE ON organization_memberships
    FOR EACH ROW EXECUTE FUNCTION update_organization_user_count();

-- Function to update group member counts
CREATE OR REPLACE FUNCTION update_group_member_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        UPDATE user_groups 
        SET total_members = (
            SELECT COUNT(*) 
            FROM group_memberships 
            WHERE group_id = NEW.group_id 
            AND status = 'active'
        ),
        active_members = (
            SELECT COUNT(*) 
            FROM group_memberships 
            WHERE group_id = NEW.group_id 
            AND status = 'active'
            AND last_activity > NOW() - INTERVAL '30 days'
        ),
        last_activity = GREATEST(last_activity, NEW.last_activity)
        WHERE id = NEW.group_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE user_groups 
        SET total_members = (
            SELECT COUNT(*) 
            FROM group_memberships 
            WHERE group_id = OLD.group_id 
            AND status = 'active'
        ),
        active_members = (
            SELECT COUNT(*) 
            FROM group_memberships 
            WHERE group_id = OLD.group_id 
            AND status = 'active'
            AND last_activity > NOW() - INTERVAL '30 days'
        )
        WHERE id = OLD.group_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger to maintain group member counts
CREATE TRIGGER trigger_update_group_member_count
    AFTER INSERT OR UPDATE OR DELETE ON group_memberships
    FOR EACH ROW EXECUTE FUNCTION update_group_member_count();

-- =============================================================================
-- SECURITY: ROW LEVEL SECURITY POLICIES
-- =============================================================================

-- Enable RLS on all tables (commented out for now, enable as needed)
-- ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE organization_memberships ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_groups ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE group_memberships ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE external_identities ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE invitations ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (uncomment and modify as needed):
-- CREATE POLICY organization_access ON organizations FOR ALL TO authenticated_users
--     USING (id IN (
--         SELECT organization_id FROM organization_memberships 
--         WHERE user_id = current_user_id() AND status = 'active'
--     ));

-- =============================================================================
-- PERFORMANCE MONITORING VIEWS
-- =============================================================================

-- View for monitoring table sizes and growth
CREATE VIEW table_size_monitoring AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation,
    most_common_vals,
    most_common_freqs,
    histogram_bounds
FROM pg_stats
WHERE schemaname = 'public'
AND tablename IN ('organizations', 'organization_memberships', 'user_groups', 
                  'group_memberships', 'import_operations', 'external_identities', 'invitations');

-- =============================================================================
-- COMMENTS AND DOCUMENTATION
-- =============================================================================

-- Table comments
COMMENT ON TABLE organizations IS 'Multi-tenant organizations supporting enterprise features';
COMMENT ON TABLE organization_memberships IS 'User-organization relationships with roles and permissions';
COMMENT ON TABLE user_groups IS 'Content creator groups within organizations';
COMMENT ON TABLE group_memberships IS 'User-group relationships with learning progress tracking';
COMMENT ON TABLE import_operations IS 'Bulk import job tracking with detailed progress and error handling';
COMMENT ON TABLE external_identities IS 'SSO and external identity provider mappings';
COMMENT ON TABLE invitations IS 'Pending invitations with onboarding progress tracking';

-- Key column comments
COMMENT ON COLUMN organizations.slug IS 'URL-friendly unique identifier for organization routing';
COMMENT ON COLUMN organizations.settings IS 'JSONB: {branding: {}, features: {}, integrations: {}}';
COMMENT ON COLUMN user_groups.features_enabled IS 'JSONB: Feature flags specific to this group';
COMMENT ON COLUMN group_memberships.progress_data IS 'JSONB: Learning progress, completion status, assessments';
COMMENT ON COLUMN import_operations.error_summary IS 'JSONB: {error_type: count} summary for quick analysis';
COMMENT ON COLUMN external_identities.provider_metadata IS 'JSONB: Additional user data from external provider';
COMMENT ON COLUMN invitations.onboarding_progress IS 'JSONB: Step-by-step onboarding completion tracking';