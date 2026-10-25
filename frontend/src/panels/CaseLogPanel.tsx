import React, { useEffect, useState, useMemo } from 'react';
import { Database, Download, Search, ChevronLeft, ChevronRight, ArrowUpDown, CheckCircle2, AlertCircle } from 'lucide-react';
import { StudyCase } from '../types';
import { API_BASE } from '../services/api';
import { Table, Column } from '../components/ui/Table';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

interface CaseLogPanelProps {
  onSelectCase: (studyId: string) => void;
}

export const CaseLogPanel: React.FC<CaseLogPanelProps> = ({ onSelectCase }) => {
  const [cases, setCases] = useState<StudyCase[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filtering & Search
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [qcFilter, setQcFilter] = useState<string>('ALL');

  // Sorting
  const [sortField, setSortField] = useState<keyof StudyCase>('timestamp');
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  // Pagination
  const [page, setPage] = useState<number>(1);
  const pageSize = 10;

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/cases`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setCases(data.cases || []);
        setError(null);
      })
      .catch((err) => {
        setError(err.message);
        // Fallback reference cases
        setCases([
          {
            study_id: 'CASE-DEMO-LIDC-0087',
            patient_id: 'LIDC-IDRI-0087',
            modality: 'CT',
            slice_count: 184,
            qc_status: 'PASS',
            nodule_count: 1,
            timestamp: '2026-09-22 12:30:00'
          },
          {
            study_id: 'CASE-DEMO-LIDC-0124',
            patient_id: 'LIDC-IDRI-0124',
            modality: 'CT',
            slice_count: 220,
            qc_status: 'PASS',
            nodule_count: 2,
            timestamp: '2026-09-21 16:45:10'
          },
          {
            study_id: 'CASE-NLST-4402',
            patient_id: 'NLST-44029',
            modality: 'CT',
            slice_count: 196,
            qc_status: 'PASS',
            nodule_count: 1,
            timestamp: '2026-09-20 09:15:22'
          }
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  // Filtered and Sorted
  const filteredCases = useMemo(() => {
    return cases.filter(c => {
      const q = searchQuery.toLowerCase();
      const matchSearch = !q || 
        c.study_id.toLowerCase().includes(q) || 
        c.patient_id.toLowerCase().includes(q) ||
        c.modality.toLowerCase().includes(q);
      const matchQC = qcFilter === 'ALL' || c.qc_status === qcFilter;
      return matchSearch && matchQC;
    });
  }, [cases, searchQuery, qcFilter]);

  const sortedCases = useMemo(() => {
    return [...filteredCases].sort((a, b) => {
      const valA = a[sortField];
      const valB = b[sortField];
      if (valA === valB) return 0;
      if (valA === undefined) return 1;
      if (valB === undefined) return -1;
      if (valA < valB) return sortAsc ? -1 : 1;
      return sortAsc ? 1 : -1;
    });
  }, [filteredCases, sortField, sortAsc]);

  const totalPages = Math.max(1, Math.ceil(sortedCases.length / pageSize));
  const paginatedCases = useMemo(() => {
    const start = (page - 1) * pageSize;
    return sortedCases.slice(start, start + pageSize);
  }, [sortedCases, page, pageSize]);

  const handleSort = (field: keyof StudyCase) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(true);
    }
  };

  const handleExportCSV = () => {
    if (filteredCases.length === 0) return;
    const headers = ['Study ID', 'Patient ID', 'Modality', 'Slice Count', 'QC Status', 'Nodule Count', 'Timestamp'];
    const rows = filteredCases.map(c => [
      c.study_id,
      c.patient_id,
      c.modality,
      c.slice_count,
      c.qc_status,
      c.nodule_count,
      c.timestamp
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `case_log_export_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const columns: Column<StudyCase>[] = [
    {
      header: 'Study ID',
      accessor: (row) => (
        <span className="mono" style={{ fontWeight: 600, color: 'var(--accent-color)' }}>
          {row.study_id}
        </span>
      ),
      width: '25%'
    },
    {
      header: 'Patient ID',
      accessor: (row) => <span className="mono">{row.patient_id}</span>,
      width: '20%'
    },
    {
      header: 'Modality',
      accessor: 'modality',
      width: '10%'
    },
    {
      header: 'Slices',
      accessor: (row) => <span className="tabular-nums">{row.slice_count}</span>,
      align: 'right',
      width: '10%'
    },
    {
      header: 'QC Status',
      accessor: (row) => (
        <Badge variant={row.qc_status === 'PASS' ? 'pass' : 'warning'}>
          {row.qc_status}
        </Badge>
      ),
      align: 'center',
      width: '12%'
    },
    {
      header: 'Nodules',
      accessor: (row) => <span className="tabular-nums">{row.nodule_count}</span>,
      align: 'right',
      width: '10%'
    },
    {
      header: 'Study Date',
      accessor: (row) => <span className="tabular-nums" style={{ color: 'var(--text-muted)' }}>{row.timestamp}</span>,
      align: 'right',
      width: '13%'
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '12px' }}>
      {/* Header controls bar */}
      <div className="panel" style={{ padding: '10px 14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={16} color="var(--accent-color)" />
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Clinical Case Log & Cohort Archive
            </span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              ({filteredCases.length} records)
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={14} color="var(--text-muted)" style={{ position: 'absolute', left: '8px' }} />
              <input
                type="text"
                value={searchQuery}
                onChange={e => { setSearchQuery(e.target.value); setPage(1); }}
                placeholder="Filter study or patient ID..."
                style={{ paddingLeft: '28px', width: '220px', height: '28px', fontSize: '12px' }}
                aria-label="Filter cases"
              />
            </div>

            <select
              value={qcFilter}
              onChange={e => { setQcFilter(e.target.value); setPage(1); }}
              style={{ height: '28px', fontSize: '12px' }}
              aria-label="Filter by QC status"
            >
              <option value="ALL">All QC</option>
              <option value="PASS">Passed QC</option>
              <option value="WARNING">Review QC</option>
            </select>

            <Button
              variant="secondary"
              onClick={handleExportCSV}
              disabled={filteredCases.length === 0}
              icon={<Download size={13} />}
            >
              Export CSV
            </Button>
          </div>
        </div>
      </div>

      {/* Main Table Panel */}
      <div className="panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {loading ? (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
              Loading clinical cases...
            </div>
          ) : (
            <Table
              columns={columns}
              data={paginatedCases}
              keyExtractor={(item) => item.study_id}
              emptyMessage={error ? `Error: ${error}` : 'No matching studies in local archive.'}
              onRowClick={(item) => onSelectCase(item.study_id)}
            />
          )}
        </div>

        {/* Pagination Bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 14px',
          borderTop: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-surface)',
          fontSize: '12px'
        }}>
          <span style={{ color: 'var(--text-muted)' }}>
            Page <strong className="tabular-nums" style={{ color: 'var(--text-primary)' }}>{page}</strong> of <span className="tabular-nums">{totalPages}</span>
          </span>

          <div style={{ display: 'flex', gap: '4px' }}>
            <Button
              variant="secondary"
              disabled={page <= 1}
              onClick={() => setPage(p => Math.max(1, p - 1))}
              aria-label="Previous page"
            >
              <ChevronLeft size={14} />
            </Button>
            <Button
              variant="secondary"
              disabled={page >= totalPages}
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              aria-label="Next page"
            >
              <ChevronRight size={14} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
