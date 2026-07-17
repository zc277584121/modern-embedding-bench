"""Local fixture data for agent procedural tool-memory retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolParameter:
    """A sanitized tool parameter description."""

    name: str
    type: str
    description: str
    default: str


@dataclass(frozen=True)
class ToolDocument:
    """An invented tool card used as a retrieval document."""

    doc_id: str
    name: str
    description: str
    parameters: tuple[ToolParameter, ...]
    source_dataset: str
    source_split: str
    source_id: str


@dataclass(frozen=True)
class ToolQuery:
    """A procedural-memory retrieval query with curated negatives."""

    query_id: str
    text: str
    positive_doc_id: str
    slice: str
    hard_negative_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class AgentProceduralToolMemoryFixture:
    """Complete local smoke fixture."""

    queries: tuple[ToolQuery, ...]
    documents: tuple[ToolDocument, ...]
    source_datasets: tuple[str, ...]
    license_audit_status: str


SOURCE_DATASET = "local_invented_tool_memory_fixture"
SOURCE_SPLIT = "smoke"
LICENSE_AUDIT_STATUS = "local_invented_sanitized_no_external_sources"


def _param(name: str, type_: str, description: str, default: str = "none") -> ToolParameter:
    return ToolParameter(name=name, type=type_, description=description, default=default)


DOCUMENTS: tuple[ToolDocument, ...] = (
    ToolDocument(
        doc_id="tool_archive_day_lookup",
        name="ArchiveDayLookup",
        description="Checks whether a public page had an archived snapshot on a requested calendar date.",
        parameters=(
            _param("page_ref", "string", "Public page identifier supplied by the caller"),
            _param("target_date", "date", "Calendar date to inspect"),
            _param("match_mode", "enum", "Exact day or closest prior snapshot", "exact"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc001",
    ),
    ToolDocument(
        doc_id="tool_archive_range_list",
        name="ArchiveRangeList",
        description="Lists archived snapshots across a date range without judging one exact date.",
        parameters=(
            _param("page_ref", "string", "Public page identifier supplied by the caller"),
            _param("start_date", "date", "First date in the range"),
            _param("end_date", "date", "Last date in the range"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc002",
    ),
    ToolDocument(
        doc_id="tool_page_freshness_score",
        name="PageFreshnessScore",
        description="Scores how recently a public page changed compared with its previous known revision.",
        parameters=(
            _param("page_ref", "string", "Public page identifier supplied by the caller"),
            _param("lookback_days", "integer", "Number of days to compare"),
            _param("change_floor", "float", "Minimum normalized change to report", "0.10"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc003",
    ),
    ToolDocument(
        doc_id="tool_bacterial_growth_projection",
        name="BacterialGrowthProjection",
        description="Forecasts population after elapsed time from an initial count, growth rate, and doubling time.",
        parameters=(
            _param("initial_population", "integer", "Starting population count"),
            _param("growth_rate", "float", "Growth-rate multiplier per interval"),
            _param("elapsed_minutes", "integer", "Minutes to forecast"),
            _param("doubling_minutes", "integer", "Minutes per doubling interval"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc004",
    ),
    ToolDocument(
        doc_id="tool_colony_plate_counter",
        name="ColonyPlateCounter",
        description="Estimates colony counts from plate observations and dilution metadata.",
        parameters=(
            _param("observed_colonies", "integer", "Visible colony count"),
            _param("dilution_factor", "float", "Dilution multiplier"),
            _param("plate_area", "float", "Area inspected on the plate"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc005",
    ),
    ToolDocument(
        doc_id="tool_growth_curve_fit",
        name="GrowthCurveFit",
        description="Fits a curve to multiple observed population readings and reports model coefficients.",
        parameters=(
            _param("time_points", "list[integer]", "Observation times"),
            _param("population_counts", "list[integer]", "Observed populations"),
            _param("model_family", "enum", "Curve family to fit", "logistic"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc006",
    ),
    ToolDocument(
        doc_id="tool_csv_profile_guard",
        name="CsvProfileGuard",
        description="Profiles a delimited file for row count, null spikes, duplicate keys, and type drift.",
        parameters=(
            _param("file_label", "string", "Local fixture file label"),
            _param("key_columns", "list[string]", "Columns that should be unique together"),
            _param("null_threshold", "float", "Allowed null fraction before warning", "0.05"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc007",
    ),
    ToolDocument(
        doc_id="tool_csv_join_preview",
        name="CsvJoinPreview",
        description="Previews join cardinality and unmatched rows before combining two delimited files.",
        parameters=(
            _param("left_file_label", "string", "Left local fixture file label"),
            _param("right_file_label", "string", "Right local fixture file label"),
            _param("join_keys", "list[string]", "Columns used for matching"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc008",
    ),
    ToolDocument(
        doc_id="tool_column_histogram",
        name="ColumnHistogram",
        description="Builds compact histograms for selected numeric columns in a delimited file.",
        parameters=(
            _param("file_label", "string", "Local fixture file label"),
            _param("columns", "list[string]", "Numeric columns to summarize"),
            _param("bucket_count", "integer", "Number of histogram buckets", "20"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc009",
    ),
    ToolDocument(
        doc_id="tool_schema_contract_diff",
        name="SchemaContractDiff",
        description="Compares two schema versions and highlights removed fields, renamed fields, and type changes.",
        parameters=(
            _param("old_schema_ref", "string", "Previous schema identifier"),
            _param("new_schema_ref", "string", "Candidate schema identifier"),
            _param("strict_types", "boolean", "Treat compatible widening as a warning instead of a pass", "true"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc010",
    ),
    ToolDocument(
        doc_id="tool_schema_sample_mapper",
        name="SchemaSampleMapper",
        description="Maps sample records onto a schema and reports fields that do not bind cleanly.",
        parameters=(
            _param("schema_ref", "string", "Schema identifier"),
            _param("sample_ref", "string", "Sample record fixture identifier"),
            _param("max_examples", "integer", "Maximum failing examples to retain", "5"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc011",
    ),
    ToolDocument(
        doc_id="tool_schema_doc_outline",
        name="SchemaDocOutline",
        description="Creates a human-readable outline from a schema with field descriptions and examples.",
        parameters=(
            _param("schema_ref", "string", "Schema identifier"),
            _param("include_examples", "boolean", "Include sanitized examples", "true"),
            _param("section_depth", "integer", "Heading depth for nested fields", "2"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc012",
    ),
    ToolDocument(
        doc_id="tool_queue_deadline_tuner",
        name="QueueDeadlineTuner",
        description="Recommends worker deadline and retry-window settings from queue latency percentiles.",
        parameters=(
            _param("queue_name", "string", "Queue identifier"),
            _param("p95_latency_ms", "integer", "Observed p95 latency"),
            _param("max_attempts", "integer", "Maximum delivery attempts"),
            _param("safety_margin_ms", "integer", "Extra margin to add", "250"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc013",
    ),
    ToolDocument(
        doc_id="tool_queue_depth_alert",
        name="QueueDepthAlert",
        description="Builds alert thresholds for backlog depth and sustained queue growth.",
        parameters=(
            _param("queue_name", "string", "Queue identifier"),
            _param("baseline_depth", "integer", "Normal backlog depth"),
            _param("growth_minutes", "integer", "Sustained growth window"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc014",
    ),
    ToolDocument(
        doc_id="tool_worker_batch_sizer",
        name="WorkerBatchSizer",
        description="Suggests worker batch size from item cost, memory limit, and target throughput.",
        parameters=(
            _param("item_cost_ms", "integer", "Average processing time per item"),
            _param("memory_limit_mb", "integer", "Worker memory limit"),
            _param("target_items_per_minute", "integer", "Desired throughput"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc015",
    ),
    ToolDocument(
        doc_id="tool_log_redaction_review",
        name="LogRedactionReview",
        description=(
            "Scans log samples for credential-like fields, session-like identifiers, "
            "and private network labels."
        ),
        parameters=(
            _param("log_sample_ref", "string", "Local sanitized log sample label"),
            _param("field_allowlist", "list[string]", "Fields allowed to remain visible"),
            _param("severity_floor", "enum", "Minimum finding severity", "medium"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc016",
    ),
    ToolDocument(
        doc_id="tool_log_volume_rollup",
        name="LogVolumeRollup",
        description="Aggregates log volume by component, severity, and time bucket.",
        parameters=(
            _param("log_sample_ref", "string", "Local sanitized log sample label"),
            _param("bucket_minutes", "integer", "Aggregation bucket size", "15"),
            _param("group_fields", "list[string]", "Fields used for grouping"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc017",
    ),
    ToolDocument(
        doc_id="tool_error_signature_cluster",
        name="ErrorSignatureCluster",
        description="Clusters repeated error messages by normalized signature and component.",
        parameters=(
            _param("log_sample_ref", "string", "Local sanitized log sample label"),
            _param("similarity_floor", "float", "Minimum signature similarity", "0.82"),
            _param("min_occurrences", "integer", "Smallest cluster to report", "3"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc018",
    ),
    ToolDocument(
        doc_id="tool_retry_backoff_designer",
        name="RetryBackoffDesigner",
        description="Designs retry attempts, exponential backoff, jitter, and maximum wait for transient operations.",
        parameters=(
            _param("base_delay_ms", "integer", "Initial retry delay"),
            _param("max_delay_ms", "integer", "Largest retry delay"),
            _param("attempts", "integer", "Number of retry attempts"),
            _param("jitter", "enum", "Jitter strategy", "full"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc019",
    ),
    ToolDocument(
        doc_id="tool_circuit_breaker_plan",
        name="CircuitBreakerPlan",
        description="Creates open, half-open, and close thresholds for protecting a flaky dependency.",
        parameters=(
            _param("failure_rate", "float", "Observed failure fraction"),
            _param("sample_window", "integer", "Number of recent requests to inspect"),
            _param("probe_count", "integer", "Half-open probe count", "5"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc020",
    ),
    ToolDocument(
        doc_id="tool_timeout_budget_split",
        name="TimeoutBudgetSplit",
        description="Splits an end-to-end timeout budget across sequential service calls.",
        parameters=(
            _param("total_budget_ms", "integer", "End-to-end timeout budget"),
            _param("service_steps", "list[string]", "Ordered service call labels"),
            _param("reserve_ms", "integer", "Budget reserved for caller overhead", "100"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc021",
    ),
    ToolDocument(
        doc_id="tool_vector_index_recipe",
        name="VectorIndexRecipe",
        description=(
            "Chooses vector index parameters from corpus size, vector dimension, "
            "recall target, and update rate."
        ),
        parameters=(
            _param("corpus_size", "integer", "Number of vectors expected"),
            _param("dimensions", "integer", "Embedding vector dimension"),
            _param("recall_target", "float", "Target recall fraction"),
            _param("updates_per_hour", "integer", "Expected update rate"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc022",
    ),
    ToolDocument(
        doc_id="tool_vector_quantization_check",
        name="VectorQuantizationCheck",
        description="Estimates storage reduction and recall risk from scalar or product quantization choices.",
        parameters=(
            _param("dimensions", "integer", "Embedding vector dimension"),
            _param("quantization_mode", "enum", "Quantization strategy", "scalar"),
            _param("sample_size", "integer", "Number of vectors in the validation sample"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc023",
    ),
    ToolDocument(
        doc_id="tool_embedding_batch_planner",
        name="EmbeddingBatchPlanner",
        description="Plans embedding batch size from text length, provider limit, and desired throughput.",
        parameters=(
            _param("avg_text_chars", "integer", "Average input length in characters"),
            _param("provider_limit", "integer", "Maximum items per request"),
            _param("target_items_per_minute", "integer", "Desired throughput"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc024",
    ),
    ToolDocument(
        doc_id="tool_locale_business_calendar",
        name="LocaleBusinessCalendar",
        description="Computes business days and observed holidays for a named locale and date range.",
        parameters=(
            _param("locale", "string", "Locale code or market label"),
            _param("start_date", "date", "First date in the range"),
            _param("end_date", "date", "Last date in the range"),
            _param("include_observed", "boolean", "Include observed holiday shifts", "true"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc025",
    ),
    ToolDocument(
        doc_id="tool_timezone_overlap",
        name="TimezoneOverlap",
        description="Finds overlapping working hours between two locations for scheduling.",
        parameters=(
            _param("left_locale", "string", "First location or locale label"),
            _param("right_locale", "string", "Second location or locale label"),
            _param("workday_hours", "string", "Working-hour window", "09:00-17:00"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc026",
    ),
    ToolDocument(
        doc_id="tool_fiscal_period_mapper",
        name="FiscalPeriodMapper",
        description="Maps calendar dates to fiscal year, quarter, period, and week labels.",
        parameters=(
            _param("fiscal_year_start", "date", "First date of the fiscal year"),
            _param("dates", "list[date]", "Dates to map"),
            _param("week_rule", "enum", "Week numbering rule", "nearest"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc027",
    ),
    ToolDocument(
        doc_id="tool_rollout_blast_radius",
        name="RolloutBlastRadius",
        description="Estimates affected users and services for a feature rollout percentage by segment.",
        parameters=(
            _param("feature_key", "string", "Feature label"),
            _param("rollout_percent", "float", "Percent of eligible traffic"),
            _param("segments", "list[string]", "Segments included in rollout"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc028",
    ),
    ToolDocument(
        doc_id="tool_experiment_sample_size",
        name="ExperimentSampleSize",
        description="Calculates sample size for an experiment from baseline rate, effect size, and power.",
        parameters=(
            _param("baseline_rate", "float", "Current conversion or success rate"),
            _param("minimum_effect", "float", "Smallest detectable effect"),
            _param("power", "float", "Desired statistical power", "0.80"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc029",
    ),
    ToolDocument(
        doc_id="tool_flag_dependency_graph",
        name="FlagDependencyGraph",
        description="Lists dependent feature flags and prerequisites before enabling a feature.",
        parameters=(
            _param("feature_key", "string", "Feature label"),
            _param("environment", "string", "Deployment environment label"),
            _param("include_disabled", "boolean", "Include currently disabled dependencies", "false"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc030",
    ),
    ToolDocument(
        doc_id="tool_incident_timeline_bundle",
        name="IncidentTimelineBundle",
        description="Builds an incident timeline from alert events, deploy markers, and operator notes.",
        parameters=(
            _param("incident_ref", "string", "Incident identifier"),
            _param("start_time", "datetime", "Beginning of the timeline window"),
            _param("end_time", "datetime", "End of the timeline window"),
            _param("include_deploys", "boolean", "Include deployment markers", "true"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc031",
    ),
    ToolDocument(
        doc_id="tool_alert_route_audit",
        name="AlertRouteAudit",
        description="Checks alert routing rules, escalation paths, and muted windows for an on-call service.",
        parameters=(
            _param("service_name", "string", "Service label"),
            _param("severity", "enum", "Alert severity to audit", "page"),
            _param("lookback_days", "integer", "Recent days to inspect", "14"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc032",
    ),
    ToolDocument(
        doc_id="tool_postmortem_action_tracker",
        name="PostmortemActionTracker",
        description="Summarizes follow-up action items, owners, and due dates from incident review notes.",
        parameters=(
            _param("review_ref", "string", "Incident review label"),
            _param("status_filter", "enum", "Action item status to include", "open"),
            _param("owner_group", "string", "Optional owner group filter", "none"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc033",
    ),
    ToolDocument(
        doc_id="tool_dependency_license_matrix",
        name="DependencyLicenseMatrix",
        description="Audits dependency names, versions, licenses, and policy compatibility.",
        parameters=(
            _param("manifest_ref", "string", "Dependency manifest label"),
            _param("policy_ref", "string", "License policy label"),
            _param("include_transitive", "boolean", "Inspect transitive dependencies", "true"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc034",
    ),
    ToolDocument(
        doc_id="tool_dependency_update_window",
        name="DependencyUpdateWindow",
        description="Finds dependency update candidates that fit a maintenance window and compatibility rule.",
        parameters=(
            _param("manifest_ref", "string", "Dependency manifest label"),
            _param("maintenance_hours", "integer", "Available maintenance window"),
            _param("compatibility_floor", "enum", "Minimum acceptable compatibility", "minor"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc035",
    ),
    ToolDocument(
        doc_id="tool_package_age_report",
        name="PackageAgeReport",
        description="Reports package age, stale versions, and release cadence for dependencies.",
        parameters=(
            _param("manifest_ref", "string", "Dependency manifest label"),
            _param("stale_days", "integer", "Age threshold for stale packages", "365"),
            _param("include_prerelease", "boolean", "Include prerelease versions", "false"),
        ),
        source_dataset=SOURCE_DATASET,
        source_split=SOURCE_SPLIT,
        source_id="doc036",
    ),
)


QUERIES: tuple[ToolQuery, ...] = (
    ToolQuery(
        query_id="q001",
        text="Find the tool that checks whether a public page had an archived snapshot on one exact date.",
        positive_doc_id="tool_archive_day_lookup",
        slice="near_name",
        hard_negative_doc_ids=(
            "tool_archive_range_list",
            "tool_page_freshness_score",
            "tool_locale_business_calendar",
        ),
    ),
    ToolQuery(
        query_id="q002",
        text=(
            "Retrieve the procedure for forecasting bacteria population from starting count, "
            "growth rate, elapsed minutes, and doubling time."
        ),
        positive_doc_id="tool_bacterial_growth_projection",
        slice="parameter_constraints",
        hard_negative_doc_ids=(
            "tool_colony_plate_counter",
            "tool_growth_curve_fit",
            "tool_experiment_sample_size",
        ),
    ),
    ToolQuery(
        query_id="q003",
        text="Select a tool to profile a delimited file for row counts, null spikes, duplicate keys, and type drift.",
        positive_doc_id="tool_csv_profile_guard",
        slice="category_collision",
        hard_negative_doc_ids=(
            "tool_csv_join_preview",
            "tool_column_histogram",
            "tool_schema_sample_mapper",
        ),
    ),
    ToolQuery(
        query_id="q004",
        text=(
            "Choose the memory entry that compares two schema versions for removed fields, "
            "renamed fields, and changed types."
        ),
        positive_doc_id="tool_schema_contract_diff",
        slice="single_tool",
        hard_negative_doc_ids=(
            "tool_schema_sample_mapper",
            "tool_schema_doc_outline",
            "tool_csv_profile_guard",
        ),
    ),
    ToolQuery(
        query_id="q005",
        text="Find the procedure that recommends queue worker deadlines and retry windows from latency percentiles.",
        positive_doc_id="tool_queue_deadline_tuner",
        slice="parameter_constraints",
        hard_negative_doc_ids=(
            "tool_queue_depth_alert",
            "tool_worker_batch_sizer",
            "tool_retry_backoff_designer",
        ),
    ),
    ToolQuery(
        query_id="q006",
        text=(
            "Retrieve the tool that reviews log samples for credential-like fields and "
            "private network labels that need redaction."
        ),
        positive_doc_id="tool_log_redaction_review",
        slice="category_collision",
        hard_negative_doc_ids=(
            "tool_log_volume_rollup",
            "tool_error_signature_cluster",
            "tool_dependency_license_matrix",
        ),
    ),
    ToolQuery(
        query_id="q007",
        text="Pick the tool for designing retry attempts with exponential backoff, jitter, and a maximum wait.",
        positive_doc_id="tool_retry_backoff_designer",
        slice="near_name",
        hard_negative_doc_ids=(
            "tool_circuit_breaker_plan",
            "tool_timeout_budget_split",
            "tool_queue_deadline_tuner",
        ),
    ),
    ToolQuery(
        query_id="q008",
        text=(
            "Find guidance for choosing vector index parameters from corpus size, "
            "dimensions, recall target, and update rate."
        ),
        positive_doc_id="tool_vector_index_recipe",
        slice="parameter_constraints",
        hard_negative_doc_ids=(
            "tool_vector_quantization_check",
            "tool_embedding_batch_planner",
            "tool_csv_profile_guard",
        ),
    ),
    ToolQuery(
        query_id="q009",
        text=(
            "Select the calendar utility that computes business days and observed holidays "
            "for a locale and date range."
        ),
        positive_doc_id="tool_locale_business_calendar",
        slice="single_tool",
        hard_negative_doc_ids=(
            "tool_timezone_overlap",
            "tool_fiscal_period_mapper",
            "tool_archive_day_lookup",
        ),
    ),
    ToolQuery(
        query_id="q010",
        text=(
            "Retrieve the tool that estimates affected users and services for a feature "
            "rollout percentage by segment."
        ),
        positive_doc_id="tool_rollout_blast_radius",
        slice="category_collision",
        hard_negative_doc_ids=(
            "tool_experiment_sample_size",
            "tool_flag_dependency_graph",
            "tool_queue_depth_alert",
        ),
    ),
    ToolQuery(
        query_id="q011",
        text="Find the incident utility that builds a timeline from alert events, deploy markers, and operator notes.",
        positive_doc_id="tool_incident_timeline_bundle",
        slice="single_tool",
        hard_negative_doc_ids=(
            "tool_alert_route_audit",
            "tool_postmortem_action_tracker",
            "tool_log_volume_rollup",
        ),
    ),
    ToolQuery(
        query_id="q012",
        text=(
            "Choose the dependency audit tool that reports package names, versions, "
            "licenses, and policy compatibility."
        ),
        positive_doc_id="tool_dependency_license_matrix",
        slice="near_name",
        hard_negative_doc_ids=(
            "tool_dependency_update_window",
            "tool_package_age_report",
            "tool_schema_contract_diff",
        ),
    ),
)


def serialize_tool_document(document: ToolDocument) -> str:
    """Serialize a tool card deterministically for embedding retrieval."""
    params = "\n".join(
        f"- {param.name} ({param.type}): {param.description}. Default: {param.default}."
        for param in document.parameters
    )
    return (
        f"Tool: {document.name}\n"
        f"Description: {document.description}\n"
        "Parameters:\n"
        f"{params}\n"
        f"Source: {document.source_dataset}/{document.source_split}/{document.source_id}"
    )


def load_agent_procedural_tool_memory_fixture(
    max_queries: int | None = None,
) -> AgentProceduralToolMemoryFixture:
    """Load and validate the built-in local smoke fixture."""
    queries = QUERIES[:max_queries] if max_queries is not None else QUERIES
    fixture = AgentProceduralToolMemoryFixture(
        queries=queries,
        documents=DOCUMENTS,
        source_datasets=(SOURCE_DATASET,),
        license_audit_status=LICENSE_AUDIT_STATUS,
    )
    _validate_fixture(fixture)
    return fixture


def fixture_slice_counts(queries: tuple[ToolQuery, ...]) -> dict[str, int]:
    """Count query slices for details reporting."""
    return dict(sorted(Counter(query.slice for query in queries).items()))


def _validate_fixture(fixture: AgentProceduralToolMemoryFixture) -> None:
    doc_ids = [document.doc_id for document in fixture.documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate tool document ids in agent procedural fixture")
    doc_id_set = set(doc_ids)

    query_ids = [query.query_id for query in fixture.queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query ids in agent procedural fixture")

    for query in fixture.queries:
        if query.positive_doc_id not in doc_id_set:
            raise ValueError(f"Unknown positive doc id for {query.query_id}: {query.positive_doc_id}")
        if len(query.hard_negative_doc_ids) != 3:
            raise ValueError(f"{query.query_id} must have exactly 3 hard negatives")
        if query.positive_doc_id in query.hard_negative_doc_ids:
            raise ValueError(f"{query.query_id} includes its positive doc as a hard negative")
        if len(set(query.hard_negative_doc_ids)) != len(query.hard_negative_doc_ids):
            raise ValueError(f"{query.query_id} has duplicate hard negatives")
        missing = set(query.hard_negative_doc_ids) - doc_id_set
        if missing:
            raise ValueError(f"{query.query_id} has unknown hard negatives: {sorted(missing)}")
