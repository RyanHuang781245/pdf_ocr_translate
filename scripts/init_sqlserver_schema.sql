IF OBJECT_ID(N'dbo.document_template_boxes', N'U') IS NOT NULL
    DROP TABLE dbo.document_template_boxes;
IF OBJECT_ID(N'dbo.document_template_pages', N'U') IS NOT NULL
    DROP TABLE dbo.document_template_pages;
IF OBJECT_ID(N'dbo.document_templates', N'U') IS NOT NULL
    DROP TABLE dbo.document_templates;
IF OBJECT_ID(N'dbo.job_events', N'U') IS NOT NULL
    DROP TABLE dbo.job_events;
IF OBJECT_ID(N'dbo.job_artifacts', N'U') IS NOT NULL
    DROP TABLE dbo.job_artifacts;
IF OBJECT_ID(N'dbo.jobs', N'U') IS NOT NULL
    DROP TABLE dbo.jobs;
GO

CREATE TABLE dbo.jobs (
    job_id char(32) NOT NULL,
    job_type varchar(50) NOT NULL,
    status varchar(30) NOT NULL,
    stage varchar(50) NULL,
    progress float NOT NULL CONSTRAINT DF_jobs_progress DEFAULT (0),
    job_name nvarchar(255) NULL,
    target_lang varchar(20) NULL,
    document_mode varchar(20) NULL,
    payload_json nvarchar(max) NULL,
    error_message nvarchar(max) NULL,
    cancel_requested bit NOT NULL CONSTRAINT DF_jobs_cancel_requested DEFAULT (0),
    retry_count int NOT NULL CONSTRAINT DF_jobs_retry_count DEFAULT (0),
    worker_id varchar(100) NULL,
    started_at datetime2(6) NULL,
    completed_at datetime2(6) NULL,
    created_at datetime2(6) NOT NULL CONSTRAINT DF_jobs_created_at DEFAULT (SYSUTCDATETIME()),
    updated_at datetime2(6) NOT NULL CONSTRAINT DF_jobs_updated_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_jobs PRIMARY KEY CLUSTERED (job_id)
);
GO

CREATE TABLE dbo.job_artifacts (
    id bigint IDENTITY(1,1) NOT NULL,
    job_id char(32) NOT NULL,
    artifact_type varchar(50) NOT NULL,
    file_path nvarchar(1000) NOT NULL,
    created_at datetime2(6) NOT NULL CONSTRAINT DF_job_artifacts_created_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_job_artifacts PRIMARY KEY CLUSTERED (id),
    CONSTRAINT FK_job_artifacts_jobs FOREIGN KEY (job_id) REFERENCES dbo.jobs(job_id)
);
GO

CREATE TABLE dbo.job_events (
    id bigint IDENTITY(1,1) NOT NULL,
    job_id char(32) NOT NULL,
    event_type varchar(50) NOT NULL,
    stage varchar(50) NULL,
    message nvarchar(max) NULL,
    created_at datetime2(6) NOT NULL CONSTRAINT DF_job_events_created_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_job_events PRIMARY KEY CLUSTERED (id),
    CONSTRAINT FK_job_events_jobs FOREIGN KEY (job_id) REFERENCES dbo.jobs(job_id)
);
GO

CREATE TABLE dbo.document_templates (
    template_id char(32) NOT NULL,
    name nvarchar(255) NOT NULL CONSTRAINT DF_document_templates_name DEFAULT (N''),
    display_name nvarchar(255) NULL,
    source_job_id char(32) NULL,
    status varchar(20) NOT NULL CONSTRAINT DF_document_templates_status DEFAULT ('saved'),
    created_at datetime2(6) NOT NULL CONSTRAINT DF_document_templates_created_at DEFAULT (SYSUTCDATETIME()),
    updated_at datetime2(6) NOT NULL CONSTRAINT DF_document_templates_updated_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_document_templates PRIMARY KEY CLUSTERED (template_id)
);
GO

CREATE TABLE dbo.document_template_pages (
    id bigint IDENTITY(1,1) NOT NULL,
    template_id char(32) NOT NULL,
    page_index_0based int NOT NULL,
    created_at datetime2(6) NOT NULL CONSTRAINT DF_document_template_pages_created_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_document_template_pages PRIMARY KEY CLUSTERED (id),
    CONSTRAINT FK_document_template_pages_templates
        FOREIGN KEY (template_id) REFERENCES dbo.document_templates(template_id) ON DELETE CASCADE,
    CONSTRAINT UQ_document_template_pages_template_page UNIQUE (template_id, page_index_0based)
);
GO

CREATE TABLE dbo.document_template_boxes (
    id bigint IDENTITY(1,1) NOT NULL,
    page_id bigint NOT NULL,
    x_ratio float NOT NULL,
    y_ratio float NOT NULL,
    w_ratio float NOT NULL,
    h_ratio float NOT NULL,
    text nvarchar(max) NOT NULL,
    font_size float NOT NULL,
    color varchar(20) NOT NULL,
    text_align varchar(20) NOT NULL,
    rotation int NOT NULL CONSTRAINT DF_document_template_boxes_rotation DEFAULT (0),
    no_clip bit NOT NULL CONSTRAINT DF_document_template_boxes_no_clip DEFAULT (0),
    created_at datetime2(6) NOT NULL CONSTRAINT DF_document_template_boxes_created_at DEFAULT (SYSUTCDATETIME()),
    updated_at datetime2(6) NOT NULL CONSTRAINT DF_document_template_boxes_updated_at DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_document_template_boxes PRIMARY KEY CLUSTERED (id),
    CONSTRAINT FK_document_template_boxes_pages
        FOREIGN KEY (page_id) REFERENCES dbo.document_template_pages(id) ON DELETE CASCADE
);
GO

CREATE INDEX IX_jobs_status_created_at
ON dbo.jobs (status, created_at);
GO

CREATE INDEX IX_jobs_job_type_updated_at
ON dbo.jobs (job_type, updated_at DESC);
GO

CREATE INDEX IX_jobs_cancel_requested_status
ON dbo.jobs (cancel_requested, status);
GO

CREATE INDEX IX_job_artifacts_job_id
ON dbo.job_artifacts (job_id);
GO

CREATE INDEX IX_job_events_job_id_created_at
ON dbo.job_events (job_id, created_at DESC);
GO

CREATE INDEX IX_document_templates_source_job_id
ON dbo.document_templates (source_job_id);
GO

CREATE INDEX IX_document_templates_updated_at
ON dbo.document_templates (updated_at DESC);
GO

CREATE INDEX IX_document_template_pages_template_id
ON dbo.document_template_pages (template_id);
GO

CREATE INDEX IX_document_template_boxes_page_id
ON dbo.document_template_boxes (page_id);
GO
