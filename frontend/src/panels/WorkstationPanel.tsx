import React, { useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE } from '../services/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

interface CaseRecord {
  study_id: string;
  patient_id?: string;
  modality?: string;
  qc_status?: string;
  status?: string;
  assigned_reader?: string;
  study_date?: string;
  timestamp?: string;
  is_demo?: boolean;
}

interface WorkstationPanelProps {
  onSelectStudy?: (studyId: string) => void;
}

export const WorkstationPanel: React.FC<WorkstationPanelProps> = ({ onSelectStudy }) => {
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [offline, setOffline] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const rowRefs = useRef<Array<HTMLTableRowElement | null>>([]);

  const [exampleLoading, setExampleLoading] = useState(false);
  const [exampleMessage, setExampleMessage] = useState<string | null>(null);

  const loadCases = () => {
    fetch(`${API_BASE}/cases`)
      .then(async response => {
        if (!response.ok) throw new Error('Worklist request failed');
        return response.json();
      })
      .then(data => {
        const records = Array.isArray(data.cases) ? data.cases : [];
        setCases(records);
        setSelectedId(records[0]?.study_id ?? null);
      })
      .catch(() => setOffline(true))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadCases();
  }, []);

  const handleLoadExampleScan = async () => {
    setExampleLoading(true);
    setExampleMessage(null);
    try {
      const exRes = await fetch(`${API_BASE}/examples`);
      const exData = await exRes.json();
      if (!exData.examples || exData.examples.length === 0) {
        setExampleMessage(exData.message || 'No example scans found under LUNA16_DATA_DIR.');
        return;
      }
      const targetScan = exData.examples[0];
      const postRes = await fetch(`${API_BASE}/examples`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: targetScan.filename }),
      });
      const postData = await postRes.json();
      if (!postRes.ok) throw new Error(postData.detail || 'Failed to load example scan');
      setExampleMessage(`Loaded ${targetScan.filename}`);
      loadCases();
      if (postData.study_id) {
        setSelectedId(postData.study_id);
        onSelectStudy?.(postData.study_id);
      }
    } catch (err: any) {
      setExampleMessage(err.message || 'Failed loading example scan');
    } finally {
      setExampleLoading(false);
    }
  };

  const filteredCases = useMemo(() => {
    const value = query.trim().toLowerCase();
    return cases.filter(item => !value || `${item.study_id} ${item.status || ''} ${item.modality || ''}`.toLowerCase().includes(value));
  }, [cases, query]);
  const selectedCase = filteredCases.find(item => item.study_id === selectedId) ?? filteredCases[0] ?? null;

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (!filteredCases.length) return;
    const currentIndex = Math.max(0, filteredCases.findIndex(item => item.study_id === selectedCase?.study_id));
    let nextIndex: number | null = null;
    if (event.key === 'ArrowDown') nextIndex = Math.min(filteredCases.length - 1, currentIndex + 1);
    if (event.key === 'ArrowUp') nextIndex = Math.max(0, currentIndex - 1);
    if (nextIndex !== null) {
      event.preventDefault();
      setSelectedId(filteredCases[nextIndex].study_id);
      rowRefs.current[nextIndex]?.focus();
    } else if (event.key === 'Enter' && selectedCase) {
      onSelectStudy?.(selectedCase.study_id);
    }
  };

  return (
    <section className="worklist-screen" onKeyDown={handleKeyDown} tabIndex={0} aria-label="Study worklist" style={{ display: 'flex', flexDirection: 'column', gap: '10px', height: '100%', minHeight: 0, outline: 'none', padding: '8px' }}>
      <header style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 600 }}>Study worklist</h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '12px' }}>Risk ordering, deferral, and workload estimates are not available without a verified evaluation result.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Button
            variant="secondary"
            size="sm"
            loading={exampleLoading}
            onClick={handleLoadExampleScan}
          >
            Load example scan
          </Button>
          <Badge variant="neutral">Research use only</Badge>
        </div>
      </header>
      {exampleMessage && (
        <div style={{ padding: '6px 10px', fontSize: '12px', background: 'var(--bg-hover)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-main)' }}>
          {exampleMessage}
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <label htmlFor="worklist-filter" style={{ fontSize: '12px' }}>Filter studies</label>
        <input id="worklist-filter" value={query} onChange={event => setQuery(event.target.value)} placeholder="Study ID, status, modality" style={{ minWidth: '220px' }} />
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Priority: Not available · Uncertainty: Not available</span>
      </div>

      <div className="worklist-grid" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.7fr) minmax(240px, 1fr)', gap: '8px', minHeight: 0, flex: 1 }}>
        <div style={{ overflow: 'auto', border: '1px solid var(--border)', background: 'var(--bg-panel)' }}>
          <table aria-label="Available studies">
            <thead><tr><th>Study ID</th><th>Study date</th><th>Modality</th><th>QC</th><th>Status</th><th>Assigned reader</th></tr></thead>
            <tbody>
              {loading ? <tr><td colSpan={6} role="status">Loading studies…</td></tr> : offline ? <tr><td colSpan={6} role="status">Worklist offline. Study data is not available.</td></tr> : filteredCases.length === 0 ? <tr><td colSpan={6} role="status">No studies available.</td></tr> : filteredCases.map((item, index) => (
                <tr
                  key={item.study_id}
                  ref={element => { rowRefs.current[index] = element; }}
                  tabIndex={0}
                  aria-selected={item.study_id === selectedCase?.study_id}
                  onClick={() => setSelectedId(item.study_id)}
                  onDoubleClick={() => onSelectStudy?.(item.study_id)}
                  style={{ cursor: 'pointer', background: item.study_id === selectedCase?.study_id ? 'var(--bg-active)' : undefined }}
                >
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{item.study_id}{item.is_demo && <span style={{ marginLeft: '6px', color: 'var(--status-warning)', fontWeight: 700 }}>SYNTHETIC</span>}</td>
                  <td>{item.study_date || 'Not available'}</td>
                  <td>{item.modality || 'Not available'}</td>
                  <td>{item.qc_status || 'Not available'}</td>
                  <td>{item.status || 'Not available'}</td>
                  <td>{item.assigned_reader || 'Not available'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ padding: '7px 10px', borderTop: '1px solid var(--border)', fontSize: '11px', color: 'var(--text-muted)' }}>
            Keyboard: Up/Down selects a study; Enter opens it. No study is automatically deferred or cleared.
          </div>
        </div>

        <aside style={{ border: '1px solid var(--border)', background: 'var(--bg-panel)', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }} aria-label="Selected study details">
          <h3 style={{ margin: 0, fontSize: '13px' }}>Study details</h3>
          {selectedCase ? <>
            <div><span style={{ color: 'var(--text-muted)' }}>Study ID: </span>{selectedCase.study_id}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Patient identifier: </span>{selectedCase.is_demo ? 'Synthetic' : 'Not available'}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Study date: </span>{selectedCase.study_date || 'Not available'}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Modality: </span>{selectedCase.modality || 'Not available'}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Model risk estimate: </span>Not available</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Uncertainty estimate: </span>Not available</div>
            {selectedCase.is_demo && <Badge variant="warning">SYNTHETIC DEMO DATA</Badge>}
            <button type="button" onClick={() => onSelectStudy?.(selectedCase.study_id)} style={{ alignSelf: 'flex-start', marginTop: '4px', padding: '6px 10px', border: '1px solid var(--border)', background: 'var(--bg-surface)', color: 'var(--text-primary)' }}>Open viewer</button>
          </> : <div role="status" style={{ color: 'var(--text-muted)' }}>Not available</div>}
          <div style={{ marginTop: 'auto', color: 'var(--text-muted)', fontSize: '11px' }}>Triage and workload curves: Not available. No validated operating point is configured.</div>
        </aside>
      </div>
    </section>
  );
};
