export interface Excerpt {
  prefix: string;
  match: string;
  suffix: string;
}

export interface Session {
  session_id: string;
  created_at: string;
  last_seen_at: string;
  display_name: string | null;
  pinned_backend: string | null;
  pinned_tier: string | null;
  summary: string | null;
  summary_updated_at: string | null;
  request_count: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cache_creation: number;
  total_cache_read: number;
  estimated_cost_usd: number;
  excerpt?: Excerpt | null;
}

export interface SessionSummary {
  session_id: string;
  summary: string;
  updated_at: string;
}

export interface ModelBreakdown {
  routed_model: string | null;
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation: number;
  cache_read: number;
  cost_usd: number;
}

export interface ConversationSummary {
  conversation_anchor: string | null;
  parent_conversation_anchor: string | null;
  request_count: number;
  started_at: string;
  last_request_ts: string;
  cost_usd: number;
  summary: string | null;
}

export interface SessionDetail extends Session {
  model_breakdown: ModelBreakdown[];
  conversations: ConversationSummary[];
}

export interface RequestRecord {
  id: number;
  session_id: string;
  conversation_anchor: string | null;
  request_ts: string;
  requested_model: string;
  routed_model: string | null;
  classification: string | null;
  reason_code: string | null;
  estimated_input_tokens: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_creation_tokens: number | null;
  cache_read_tokens: number | null;
  duration_ms: number | null;
  backend: string;
  status: string;
  error: string | null;
  applied: number | null;
  cost_estimate: number | null;
  model_tier: string | null;
  attempt: number;
  user_prompt_text?: string | null;
  response_text?: string | null;
  excerpt?: Excerpt | null;
}

export interface CostRow {
  name: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation: number;
  cache_read: number;
  cost_usd: number;
  cache_savings_usd: number;
}

export interface ConfigChange {
  id: number;
  ts: string;
  event_type: string;
  actor: string;
  actor_id: string;
  prev_value: string | null;
  new_value: string | null;
}

export interface Backend {
  name: string;
  active: boolean;
  available: boolean | null;
}

export interface Config {
  routing_enabled: boolean;
  auto_backend_mode: string;
  auto_backend: boolean;
  active_backend: string;
  auto_model_routing_classifier_model: string;
  auto_model_routing_long_context_threshold: number;
  auto_model_routing_affirmation_inherit: boolean;
  auto_model_routing_mode: string;
}

export interface SessionsResponse {
  items: Session[];
  total: number;
  limit: number;
  offset: number;
  q: string;
}

export interface TraceResponse {
  items: RequestRecord[];
  session_id: string;
  anchor: string | null;
  total: number;
  limit: number;
  offset: number;
  q: string;
}

export interface CostResponse {
  items: CostRow[];
  group_by: string;
  time_range: string;
}

export interface RoutingResponse {
  reason_codes: { reason_code: string; count: number }[];
  tier_transitions: { requested_tier: string; routed_tier: string; count: number }[];
  upgrade_count: number;
  downgrade_count: number;
  unchanged_count: number;
  size_forced_count: number;
  affirmation_count: number;
  cached_tier_count: number;
  original_model_distribution: Array<{ model: string; count: number }>;
  routed_model_distribution: Array<{ model: string; count: number }>;
}

export interface BackendsResponse {
  backends: Backend[];
  active: string;
  known: string[];
  modes: string[];
}

export interface ConfigChangesResponse {
  items: ConfigChange[];
}

// GET /admin/status
export interface BackendAvailability {
  name: string;
  active: boolean;
  available: boolean | null;
}

export interface SessionOverride {
  session_id: string;
  display_name: string | null;
  pinned_backend: string | null;
  pinned_tier: string | null;
}

export interface UsageWindow {
  used_tokens: number | null;
  limit_tokens: number | null;
  pct: number | null;
  reset_at: string | null;
  reset_in_secs: number | null;
  window_hours?: number | null;
  active_secs?: number | null;
}

export interface CreditsWindow {
  used_usd: number | null;
  total_usd: number | null;
  pct: number | null;
}

export interface BackendUsage {
  five_hour?: UsageWindow;
  weekly?: UsageWindow;
  credits?: CreditsWindow;
  age_secs?: number | null;
}

export interface StatusResponse {
  active_backend: string;
  auto_selection?: string | null;
  routing_enabled: boolean;
  routing_mode: string;
  classifier_model: string;
  long_context_threshold: number;
  affirmation_inherit: boolean;
  backends: BackendAvailability[];
  session_overrides: SessionOverride[];
  subscription_usage: Record<string, BackendUsage>;
}

// POST /admin/global/backend response
export interface SetBackendResponse {
  active_backend: string;
  auto_selection?: string | null;
}

// GET /admin/stats
export interface StatsRow {
  backend: string;
  model_tier: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  cost_usd: number;
  cache_savings_usd: number;
  active_time_secs: number;
}

export interface StatsBucket {
  label: string;
  rows: StatsRow[];
  subtotal: StatsRow;
}

export type StatsTotal = StatsRow;

export interface StatsResponse {
  period: string;
  backend_filter: string | null;
  buckets: StatsBucket[];
  total: StatsTotal;
}

/** Typed view of the classifier_summary_json blob stored per request. */
export interface ClassifierSummary {
  final_user_text?: string;
  prior_response_summary?: string;
  total_messages?: number;
  prior_user_messages?: number;
  prior_assistant_messages?: number;
  tool_use_count?: number;
  tool_result_count?: number;
  final_non_text_blocks?: number;
  has_images?: boolean;
  text_truncated?: boolean;
  recovered_via_walkback?: boolean;
}

// GET /admin/requests/{id}
export interface RequestDetail extends RequestRecord {
  user_prompt_text: string | null;
  response_text: string | null;
  system_prompt_sha256: string | null;
  tools_sha256: string | null;
  routing_recovered_via_walkback: number | null;
  classifier_model: string | null;
  classifier_summary_json: string | null;
  classifier_raw_response: string | null;
  classifier_confidence: number | null;
  classifier_format: string | null;
  cache_savings_usd: number | null;
  system_prompt_content: string | null;
  system_prompt_char_count: number | null;
  tools_content: string | null;
  tools_char_count: number | null;
}
