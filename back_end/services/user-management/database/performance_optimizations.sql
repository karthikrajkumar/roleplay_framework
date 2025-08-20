-- =============================================================================
-- PERFORMANCE OPTIMIZATION EXTENSIONS FOR USER MANAGEMENT SYSTEM
-- PostgreSQL Database Optimizations for Scale (100k+ users, Sub-100ms queries)
-- =============================================================================

-- Enable additional performance extensions
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"; -- Query performance monitoring
CREATE EXTENSION IF NOT EXISTS "pg_buffercache"; -- Buffer cache analysis
CREATE EXTENSION IF NOT EXISTS "pgstattuple"; -- Table and index statistics

-- =============================================================================
-- 1. CRITICAL INDEX OPTIMIZATIONS FOR SUB-100MS QUERIES
-- =============================================================================

-- High-performance composite indexes for group membership queries
DROP INDEX IF EXISTS idx_group_memberships_group;
DROP INDEX IF EXISTS idx_group_memberships_user;

-- Optimized group membership queries (most critical performance path)
CREATE INDEX CONCURRENTLY idx_group_memberships_user_active_role 
    ON group_memberships(user_id, status, role) 
    WHERE status = 'active';

CREATE INDEX CONCURRENTLY idx_group_memberships_group_active_role 
    ON group_memberships(group_id, status, role) 
    WHERE status = 'active';

-- Permission check optimization - covering index for fast lookups
CREATE INDEX CONCURRENTLY idx_group_memberships_permissions_lookup 
    ON group_memberships(user_id, group_id, role, status) 
    INCLUDE (access_level, progress_data)
    WHERE status = 'active';

-- Organization membership fast lookup
CREATE INDEX CONCURRENTLY idx_org_memberships_user_active_permissions 
    ON organization_memberships(user_id, organization_id, status) 
    INCLUDE (role, permissions, access_level)
    WHERE status = 'active';

-- User groups discovery optimization
CREATE INDEX CONCURRENTLY idx_user_groups_org_visibility_active 
    ON user_groups(organization_id, visibility, status) 
    INCLUDE (name, category, max_members, total_members)
    WHERE status = 'active';

-- =============================================================================
-- 2. BULK IMPORT PERFORMANCE OPTIMIZATIONS
-- =============================================================================

-- Bulk import job monitoring and queue management
CREATE INDEX CONCURRENTLY idx_import_jobs_priority_queue 
    ON import_operations(organization_id, status, created_at) 
    WHERE status IN ('pending', 'processing', 'validating');

-- Batch processing optimization
CREATE INDEX CONCURRENTLY idx_import_batches_processing_queue 
    ON import_batches(import_job_id, status, batch_number) 
    WHERE status IN ('pending', 'processing');

-- Duplicate detection performance
CREATE INDEX CONCURRENTLY idx_duplicate_cache_email_active 
    ON duplicate_detection_cache(email_hash, source_type) 
    WHERE expires_at > NOW() OR expires_at IS NULL;

-- Fast user lookup for duplicate checking
CREATE INDEX CONCURRENTLY idx_users_bulk_import_lookup 
    ON users(email, username, status) 
    INCLUDE (id, created_at)
    WHERE status = 'active';

-- =============================================================================
-- 3. ANALYTICS QUERY OPTIMIZATIONS
-- =============================================================================

-- Organization analytics fast aggregation
CREATE INDEX CONCURRENTLY idx_org_memberships_analytics 
    ON organization_memberships(organization_id, status, joined_at) 
    INCLUDE (role, last_activity);

-- Group activity analytics
CREATE INDEX CONCURRENTLY idx_group_memberships_analytics 
    ON group_memberships(group_id, status, joined_at, last_activity) 
    INCLUDE (role, participation_score);

-- User activity tracking
CREATE INDEX CONCURRENTLY idx_users_activity_analytics 
    ON users(subscription_tier, status, last_activity) 
    INCLUDE (created_at, total_conversations);

-- =============================================================================
-- 4. PARTIAL INDEXES FOR MEMORY EFFICIENCY
-- =============================================================================

-- Only index active invitations
CREATE INDEX CONCURRENTLY idx_invitations_active_processing 
    ON invitations(organization_id, email, status, expires_at) 
    WHERE status IN ('pending', 'sent', 'opened') AND expires_at > NOW();

-- Only index recent sessions
CREATE INDEX CONCURRENTLY idx_user_sessions_recent_active 
    ON user_sessions(user_id, is_active, last_activity) 
    WHERE is_active = TRUE AND last_activity > NOW() - INTERVAL '30 days';

-- Only index tokens that haven't expired
CREATE INDEX CONCURRENTLY idx_auth_tokens_active_valid 
    ON auth_tokens(user_id, token_type, is_active) 
    WHERE is_active = TRUE AND expires_at > NOW();

-- =============================================================================
-- 5. JSONB OPTIMIZATIONS FOR SETTINGS AND METADATA
-- =============================================================================

-- GIN indexes with specific operator classes for better performance
DROP INDEX IF EXISTS idx_organizations_settings;
DROP INDEX IF EXISTS idx_user_groups_settings;
DROP INDEX IF EXISTS idx_group_memberships_progress;

-- Optimized JSONB indexes with jsonb_path_ops for exact key lookups
CREATE INDEX CONCURRENTLY idx_organizations_settings_pathops 
    ON organizations USING gin(settings jsonb_path_ops);

CREATE INDEX CONCURRENTLY idx_user_groups_settings_pathops 
    ON user_groups USING gin(group_settings jsonb_path_ops);

CREATE INDEX CONCURRENTLY idx_group_memberships_progress_pathops 
    ON group_memberships USING gin(progress_data jsonb_path_ops);

-- Specific JSONB key indexes for common queries
CREATE INDEX CONCURRENTLY idx_organizations_feature_flags 
    ON organizations USING gin((feature_flags->'features')) 
    WHERE feature_flags IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_group_memberships_completion_status 
    ON group_memberships USING gin((progress_data->'completion')) 
    WHERE progress_data IS NOT NULL;

-- =============================================================================
-- 6. TEXT SEARCH OPTIMIZATION
-- =============================================================================

-- Full-text search indexes for user and group discovery
CREATE INDEX CONCURRENTLY idx_users_fulltext_search 
    ON users USING gin(to_tsvector('english', 
        COALESCE(username, '') || ' ' || 
        COALESCE(email, '') || ' ' || 
        COALESCE(profile_data->>'first_name', '') || ' ' || 
        COALESCE(profile_data->>'last_name', '')
    )) WHERE status = 'active';

CREATE INDEX CONCURRENTLY idx_user_groups_fulltext_search 
    ON user_groups USING gin(to_tsvector('english', 
        COALESCE(name, '') || ' ' || 
        COALESCE(description, '') || ' ' || 
        COALESCE(category, '') || ' ' || 
        array_to_string(tags, ' ')
    )) WHERE status = 'active';

-- =============================================================================
-- 7. CONSTRAINT OPTIMIZATIONS
-- =============================================================================

-- Add check constraints for data integrity and query optimization
ALTER TABLE group_memberships 
ADD CONSTRAINT ck_group_memberships_progress_valid 
CHECK (
    progress_data IS NULL OR 
    jsonb_typeof(progress_data) = 'object'
);

ALTER TABLE organizations 
ADD CONSTRAINT ck_organizations_user_limits 
CHECK (total_users <= max_users * 2); -- Allow some buffer for processing

-- =============================================================================
-- 8. MATERIALIZED VIEW ENHANCEMENTS
-- =============================================================================

-- Enhanced organization usage summary with performance metrics
DROP MATERIALIZED VIEW IF EXISTS organization_usage_summary;
CREATE MATERIALIZED VIEW organization_usage_summary AS
SELECT 
    o.id as organization_id,
    o.name as organization_name,
    o.subscription_tier,
    o.max_users,
    o.max_groups,
    
    -- User metrics
    COUNT(DISTINCT om.user_id) FILTER (WHERE om.status = 'active') as active_users,
    COUNT(DISTINCT om.user_id) as total_users,
    COUNT(DISTINCT om.user_id) FILTER (WHERE om.last_activity > NOW() - INTERVAL '30 days') as monthly_active_users,
    COUNT(DISTINCT om.user_id) FILTER (WHERE om.last_activity > NOW() - INTERVAL '7 days') as weekly_active_users,
    
    -- Group metrics
    COUNT(DISTINCT ug.id) FILTER (WHERE ug.status = 'active') as active_groups,
    COUNT(DISTINCT ug.id) as total_groups,
    AVG(ug.total_members) as avg_group_size,
    
    -- Membership metrics
    COUNT(DISTINCT gm.user_id) FILTER (WHERE gm.status = 'active') as active_group_members,
    COUNT(DISTINCT gm.id) as total_memberships,
    AVG(gm.participation_score) FILTER (WHERE gm.status = 'active') as avg_participation_score,
    
    -- Invitation metrics
    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'pending') as pending_invitations,
    COUNT(DISTINCT i.id) FILTER (WHERE i.status = 'accepted') as accepted_invitations,
    
    -- Performance metrics
    ROUND((COUNT(DISTINCT om.user_id)::float / GREATEST(o.max_users, 1) * 100)::numeric, 2) as user_utilization_percent,
    ROUND((COUNT(DISTINCT ug.id)::float / GREATEST(o.max_groups, 1) * 100)::numeric, 2) as group_utilization_percent,
    
    -- Timestamps
    MAX(GREATEST(
        COALESCE(om.last_activity, om.created_at),
        COALESCE(ug.last_activity, ug.created_at),
        COALESCE(gm.last_activity, gm.created_at)
    )) as last_activity,
    NOW() as last_updated
FROM organizations o
LEFT JOIN organization_memberships om ON o.id = om.organization_id
LEFT JOIN user_groups ug ON o.id = ug.organization_id  
LEFT JOIN group_memberships gm ON ug.id = gm.group_id
LEFT JOIN invitations i ON o.id = i.organization_id
WHERE o.status = 'active'
GROUP BY o.id, o.name, o.subscription_tier, o.max_users, o.max_groups;

-- Performance index for the enhanced materialized view
CREATE UNIQUE INDEX idx_org_usage_summary_org_id ON organization_usage_summary(organization_id);
CREATE INDEX idx_org_usage_summary_tier_utilization ON organization_usage_summary(subscription_tier, user_utilization_percent);
CREATE INDEX idx_org_usage_summary_activity ON organization_usage_summary(last_activity DESC);

-- =============================================================================
-- 9. AUTOMATIC MAINTENANCE FUNCTIONS
-- =============================================================================

-- Function to refresh materialized views efficiently
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS VOID AS $$
BEGIN
    -- Refresh with minimal locking
    REFRESH MATERIALIZED VIEW CONCURRENTLY organization_usage_summary;
    
    -- Log refresh completion
    INSERT INTO system_maintenance_log (operation, completed_at, details)
    VALUES ('refresh_analytics_views', NOW(), 'Refreshed organization_usage_summary');
END;
$$ LANGUAGE plpgsql;

-- Function to clean up expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean expired invitations
    DELETE FROM invitations 
    WHERE status IN ('pending', 'sent', 'opened') 
    AND expires_at < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean old user sessions
    DELETE FROM user_sessions 
    WHERE ended_at IS NOT NULL 
    AND ended_at < NOW() - INTERVAL '90 days';
    
    -- Clean expired auth tokens
    DELETE FROM auth_tokens 
    WHERE expires_at < NOW() - INTERVAL '7 days';
    
    -- Clean duplicate detection cache
    DELETE FROM duplicate_detection_cache 
    WHERE expires_at IS NOT NULL 
    AND expires_at < NOW();
    
    -- Clean old validation errors (keep for audit)
    DELETE FROM validation_errors 
    WHERE created_at < NOW() - INTERVAL '1 year'
    AND import_job_id IN (
        SELECT id FROM import_operations 
        WHERE status = 'completed' 
        AND completed_at < NOW() - INTERVAL '1 year'
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 10. MONITORING AND STATISTICS
-- =============================================================================

-- Function to gather table statistics for monitoring
CREATE OR REPLACE FUNCTION get_table_performance_stats()
RETURNS TABLE (
    table_name TEXT,
    total_size TEXT,
    table_size TEXT,
    index_size TEXT,
    row_count BIGINT,
    seq_scan BIGINT,
    seq_tup_read BIGINT,
    idx_scan BIGINT,
    idx_tup_fetch BIGINT,
    n_tup_ins BIGINT,
    n_tup_upd BIGINT,
    n_tup_del BIGINT,
    last_vacuum TIMESTAMP,
    last_analyze TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename AS table_name,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size,
        st.n_tup_ins + st.n_tup_upd + st.n_tup_del AS row_count,
        st.seq_scan,
        st.seq_tup_read,
        st.idx_scan,
        st.idx_tup_fetch,
        st.n_tup_ins,
        st.n_tup_upd,
        st.n_tup_del,
        st.last_vacuum,
        st.last_analyze
    FROM pg_stat_user_tables st
    WHERE schemaname = 'public'
    AND tablename IN ('users', 'organizations', 'user_groups', 'group_memberships', 
                      'organization_memberships', 'import_operations', 'import_batches')
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- System maintenance log table
CREATE TABLE IF NOT EXISTS system_maintenance_log (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(100) NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_ms INTEGER,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_maintenance_log_operation ON system_maintenance_log(operation, completed_at DESC);

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON FUNCTION refresh_analytics_views() IS 'Refreshes materialized views for analytics with minimal locking impact';
COMMENT ON FUNCTION cleanup_expired_data() IS 'Removes expired data to maintain database performance';
COMMENT ON FUNCTION get_table_performance_stats() IS 'Returns comprehensive performance statistics for monitoring';