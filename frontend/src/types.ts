export interface NoduleMorphometry {
  nodule_id: string;
  centroid_zyx: number[];
  effective_diameter_mm: number;
  volume_mm3: number;
  mean_hu: number;
  std_hu: number;
  max_hu: number;
  min_hu: number;
  sphericity: number;
  spiculation_index: number;
  surface_to_volume_ratio: number;
}

export interface NoduleItem {
  nodule_id: string;
  center_zyx: number[];
  diameter_mm: number;
  volume_mm3: number;
  lobe: string;
  preliminary_risk: string;
  morphometry: NoduleMorphometry;
}

export interface StudyCase {
  study_id: string;
  patient_id: string;
  modality: string;
  slice_count: number;
  qc_status: string;
  nodule_count: number;
  timestamp: string;
}

export interface ModelBreakdownItem {
  model_id: string;
  model_name: string;
  raw_probability: number;
  calibrated_probability: number;
  uncertainty_score: number;
  uncertainty_level: 'LOW' | 'MODERATE' | 'HIGH';
  status: string;
  weight_in_ensemble: number;
  verdict: string;
}

export interface VerdictResult {
  nodule_id: string;
  ensemble_verdict: string;
  ensemble_probability: number;
  confidence_interval: number[];
  weights_used: Record<string, number>;
  disagreement_entropy: number;
  model_breakdown: ModelBreakdownItem[];
  morphometry: NoduleMorphometry;
  clinical_advisory: string;
}

export interface BenchmarkModel {
  model_name: string;
  architecture: string;
  status: string;
  auc_roc: number;
  sensitivity: number;
  specificity: number;
  ece: number;
  brier_score: number;
  notes: string;
}

export interface ModelRegistryItem {
  model_name: string;
  version: string;
  stage: string;
  architecture: string;
  input_dimension: string;
  metrics: Record<string, number>;
  created_at: string;
  notes: string;
}
