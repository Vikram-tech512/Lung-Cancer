/**
 * frontend/src/services/api.ts
 * Unified API client with automatic JWT token injection, error handling,
 * and REST communication with the FastAPI backend.
 */

const API_BASE = '/api';

export function getAuthToken(): string | null {
  return localStorage.getItem('access_token');
}

export function setAuthToken(token: string) {
  localStorage.setItem('access_token', token);
}

export function clearAuthToken() {
  localStorage.removeItem('access_token');
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {})
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`API Error ${response.status}: ${errText}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  // Auth
  login: (email: string, password: string) =>
    request<any>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password })
    }),
  getMe: () => request<any>('/auth/me'),

  // Studies & QC
  listStudies: () => request<any[]>('/studies'),
  uploadStudy: () => request<any>('/studies/upload', { method: 'POST' }),
  getQCReport: (studyId: string) => request<any>(`/studies/${studyId}/qc`),

  // Analysis Pipeline
  startAnalysis: (studyId: string) =>
    request<any>('/analysis', {
      method: 'POST',
      body: JSON.stringify({ study_id: studyId })
    }),
  getAnalysisStatus: (jobId: string) => request<any>(`/analysis/${jobId}`),

  // Viewer & XAI
  getStudyNodules: (studyId: string) => request<any>(`/viewer/${studyId}/nodules`),
  getOverlay: (lesionId: string, layerType: string) =>
    request<any>(`/viewer/overlay/${lesionId}/${layerType}`),

  // Verdict & Morphometry
  getVerdict: (lesionId: string) => request<any>(`/verdict/${lesionId}`),
  getConfidenceSignature: (lesionId: string) =>
    request<any>(`/verdict/${lesionId}/confidence-signature`),

  // Models & Benchmarks
  listModels: () => request<any[]>('/models'),

  // Experiments & Datasets
  listExperiments: () => request<any[]>('/experiments'),
  listDatasets: () => request<any[]>('/datasets'),

  // Reports
  listReports: () => request<any[]>('/reports'),

  // Admin & Domain Settings
  getTelemetry: () => request<any>('/admin/telemetry'),
  getDomainSettings: () => request<any>('/settings/domain'),
  updateDomainSettings: (data: any) =>
    request<any>('/settings/domain', {
      method: 'POST',
      body: JSON.stringify(data)
    }),

  // System
  getHealth: () => request<any>('/health')
};
