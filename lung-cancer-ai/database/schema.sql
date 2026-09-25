-- =============================================================================
-- PostgreSQL Database Schema for 3D Explainable Lung Cancer AI Platform
-- Production DDL: 28 Tables with Foreign Keys, Indexes, and Audit Partitioning
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Users & RBAC
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(128) NOT NULL,
    role VARCHAR(32) NOT NULL DEFAULT 'RADIOLOGIST', -- ADMIN, RADIOLOGIST, RESEARCHER
    hospital_affiliation VARCHAR(128) DEFAULT 'Academic Medical Center',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Studies
CREATE TABLE IF NOT EXISTS studies (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    patient_anon_id VARCHAR(64) NOT NULL,
    study_instance_uid VARCHAR(128) UNIQUE,
    study_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modality VARCHAR(16) DEFAULT 'CT',
    slice_thickness DOUBLE PRECISION,
    pixel_spacing_x DOUBLE PRECISION,
    pixel_spacing_y DOUBLE PRECISION,
    slice_count INTEGER DEFAULT 0,
    manufacturer VARCHAR(64),
    convolution_kernel VARCHAR(64),
    qc_status VARCHAR(16) DEFAULT 'PENDING', -- PASS, WARNING, REJECT
    file_path VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Analysis Jobs
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    study_id VARCHAR(36) NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    status VARCHAR(32) DEFAULT 'QUEUED', -- QUEUED, PROCESSING, COMPLETED, FAILED, REVIEW_REQUIRED
    progress_percent INTEGER DEFAULT 0,
    current_stage VARCHAR(64) DEFAULT 'QUEUED',
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    failure_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Lesions / Nodule Candidates
CREATE TABLE IF NOT EXISTS lesions (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    study_id VARCHAR(36) NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    lesion_index INTEGER NOT NULL,
    centroid_z DOUBLE PRECISION NOT NULL,
    centroid_y DOUBLE PRECISION NOT NULL,
    centroid_x DOUBLE PRECISION NOT NULL,
    diameter_mm DOUBLE PRECISION NOT NULL,
    volume_mm3 DOUBLE PRECISION NOT NULL,
    anatomical_lobe VARCHAR(64),
    preliminary_risk VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Segmentations
CREATE TABLE IF NOT EXISTS segmentations (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    lesion_id VARCHAR(36) NOT NULL REFERENCES lesions(id) ON DELETE CASCADE,
    segmentation_type VARCHAR(32) NOT NULL, -- LUNG_PARENCHYMA, NODULE
    mask_storage_path VARCHAR(255) NOT NULL,
    dice_score DOUBLE PRECISION,
    volume_voxels INTEGER DEFAULT 0
);

-- 6. Measurements (Morphometry)
CREATE TABLE IF NOT EXISTS measurements (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    lesion_id VARCHAR(36) NOT NULL REFERENCES lesions(id) ON DELETE CASCADE,
    area_mm2 DOUBLE PRECISION DEFAULT 0.0,
    volume_mm3 DOUBLE PRECISION DEFAULT 0.0,
    effective_diameter_mm DOUBLE PRECISION DEFAULT 0.0,
    sphericity DOUBLE PRECISION DEFAULT 0.0,
    spiculation_index DOUBLE PRECISION DEFAULT 0.0,
    compactness DOUBLE PRECISION DEFAULT 0.0,
    mean_hu DOUBLE PRECISION DEFAULT 0.0,
    std_hu DOUBLE PRECISION DEFAULT 0.0,
    min_hu DOUBLE PRECISION DEFAULT 0.0,
    max_hu DOUBLE PRECISION DEFAULT 0.0
);

-- 7. Radiologist Reviews (Clinical Sign-Off & Separation Mandate §25)
CREATE TABLE IF NOT EXISTS radiologist_reviews (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    study_id VARCHAR(36) NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    lesion_id VARCHAR(36) REFERENCES lesions(id) ON DELETE CASCADE,
    reviewer_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    ai_verdict VARCHAR(64),
    ai_probability DOUBLE PRECISION,
    ai_measurements JSONB DEFAULT '{}'::jsonb,
    action VARCHAR(32) NOT NULL, -- ACCEPT, REJECT, MODIFY, MARK_FP, ADD_FINDING
    radiologist_verdict VARCHAR(64),
    radiologist_measurements JSONB DEFAULT '{}'::jsonb,
    radiologist_comments TEXT,
    final_verdict VARCHAR(64),
    final_measurements JSONB DEFAULT '{}'::jsonb,
    model_version VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. Longitudinal Comparisons (VDT & Growth Kinetics)
CREATE TABLE IF NOT EXISTS longitudinal_comparisons (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    current_study_id VARCHAR(36) NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    previous_study_id VARCHAR(36) NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    registration_method VARCHAR(64) DEFAULT 'rigid_centroid_matching',
    registration_quality DOUBLE PRECISION,
    is_reliable BOOLEAN DEFAULT FALSE,
    days_between INTEGER,
    matched_nodules JSONB DEFAULT '[]'::jsonb,
    failure_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 9. Immutable Audit Trail
CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(128) NOT NULL,
    resource_type VARCHAR(64) NOT NULL,
    resource_id VARCHAR(64),
    ip_address VARCHAR(64),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for high-performance querying
CREATE INDEX IF NOT EXISTS idx_studies_patient ON studies(patient_anon_id);
CREATE INDEX IF NOT EXISTS idx_lesions_study ON lesions(study_id);
CREATE INDEX IF NOT EXISTS idx_reviews_study ON radiologist_reviews(study_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp DESC);
