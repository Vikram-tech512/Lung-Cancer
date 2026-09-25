/**
 * frontend/src/services/api.ts
 * =============================
 * Unified API Client for Lung Cancer AI Research Platform.
 * Proxies through Vite / Nginx to FastAPI backend at port 8000.
 */

const API_BASE = '/api/v2';

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {})
  };

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  // System Health
  health: () => request<{ status: string; version: string; active_models: string[]; disclaimer: string }>('/health'),

  // Case Management
  listCases: () => request<{ cases: any[] }>('/cases'),

  // DICOM Intake & Quality Gate
  uploadDicom: (formData?: FormData) => {
    return fetch(`${API_BASE}/dicom/upload`, {
      method: 'POST',
      body: formData
    }).then(res => res.json());
  },
  getQCReport: (studyId: string) => request<any>(`/dicom/${studyId}/quality`),

  // Classification & Multimodal Ensemble
  classifyNodule: (noduleId: string) => request<any>(`/classify/${noduleId}`, { method: 'POST' }),

  // 3D Morphometry & Radiomics
  getMorphometry: (noduleId: string) => request<any>(`/morphometry/${noduleId}`),

  // Explainable AI (3D Grad-CAM, Integrated Gradients, Attention Rollout)
  getXAI: (noduleId: string, method: 'gradcam3d' | 'integrated_gradients' | 'attention_rollout' = 'gradcam3d') =>
    request<any>(`/xai/${noduleId}/${method}`),

  // Benchmarking & Robustness
  getBenchmark: () => request<any>('/benchmark'),
  perturbStudy: (studyId: string, perturbationType: string, severity: number) =>
    request<any>(`/robustness/${studyId}/perturb`, {
      method: 'POST',
      body: JSON.stringify({ perturbation_type: perturbationType, severity })
    }),

  // Model Registry & Experiment Tracking
  getRegistry: () => request<{ models: any[] }>('/registry'),
  promoteModel: (modelName: string, version: string, stage: string) =>
    request<any>(`/registry/${modelName}/promote`, {
      method: 'POST',
      body: JSON.stringify({ version, stage })
    }),
  getExperiments: () => request<{ runs: any[] }>('/experiments'),
  startTraining: (modelName: string, epochs: number = 50, batchSize: number = 4, lr: number = 1e-4) => {
    const data = new FormData();
    data.append('model_name', modelName);
    data.append('epochs', epochs.toString());
    data.append('batch_size', batchSize.toString());
    data.append('learning_rate', lr.toString());
    return fetch(`${API_BASE}/training/start`, {
      method: 'POST',
      body: data
    }).then(res => res.json());
  }
};
