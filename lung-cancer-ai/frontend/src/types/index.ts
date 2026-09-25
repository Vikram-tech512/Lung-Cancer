export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'ADMIN' | 'RESEARCHER' | 'VIEWER';
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  email: string;
  role: string;
}

export interface Study {
  id: string;
  patient_id: string;
  study_instance_uid: string;
  modality: string;
  slice_count: number;
  slice_thickness: number;
  qc_status: string;
  created_at: string;
}

export interface QCReport {
  status: string;
  slice_count: number;
  slice_thickness: number;
  findings: string[];
  is_acceptable: boolean;
}

export interface Morphometry {
  lesion_id: string;
  area_mm2: number;
  volume_mm3: number;
  effective_diameter_mm: number;
  sphericity: number;
  spiculation_index: number;
  compactness: number;
  mean_hu: number;
  std_hu: number;
  min_hu: number;
  max_hu: number;
}

export interface ModelBreakdownItem {
  model_name: string;
  raw_probability: number;
  calibrated_probability: number;
  uncertainty_score: number;
  uncertainty_level: string;
  status: string;
  latency_ms: number;
}

export interface Verdict {
  lesion_id: string;
  consensus_verdict: string;
  consensus_probability: number;
  ci_lower: number;
  ci_upper: number;
  disagreement_entropy: number;
  model_breakdown: ModelBreakdownItem[];
  morphometry: Morphometry;
  clinical_advisory: string;
}

export interface AnalysisEvent {
  analysis_id: string;
  stage: string;
  progress_percent: number;
  data: Record<string, any>;
  timestamp: number;
}

export interface TelemetryData {
  timestamp: string;
  server_status: string;
  cpu_load_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  disk_free_gb: number;
  gpu_telemetry: string;
  services: Record<string, string>;
  active_jobs: number;
}

export interface DomainSettings {
  app_name: string;
  public_domain: string;
  server_ip: string;
  ssl_status: string;
}
