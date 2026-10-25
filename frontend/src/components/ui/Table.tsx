import React from 'react';

export interface Column<T> {
  header: string;
  accessor?: keyof T | ((row: T) => React.ReactNode);
  align?: 'left' | 'center' | 'right';
  className?: string;
  width?: string;
}

export interface TableProps<T = any> {
  columns?: Column<T>[];
  data?: T[];
  keyExtractor?: (item: T, index: number) => string | number;
  emptyMessage?: string;
  onRowClick?: (item: T) => void;
  selectedKey?: string | number;
  children?: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export function Table<T = any>({
  columns,
  data,
  keyExtractor,
  emptyMessage = 'No records found',
  onRowClick,
  selectedKey,
  children,
  className = '',
  style,
}: TableProps<T>) {
  // If compound children are provided (Thead, Tbody, etc.)
  if (children) {
    return (
      <div style={{ width: '100%', overflowX: 'auto', ...style }}>
        <table className={className} style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
          {children}
        </table>
      </div>
    );
  }

  // Otherwise, use declarative columns & data API
  if (!data || data.length === 0) {
    return (
      <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', ...style }}>
        {emptyMessage}
      </div>
    );
  }

  return (
    <div style={{ width: '100%', overflowX: 'auto', ...style }}>
      <table className={className} style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
        <thead>
          <tr>
            {columns?.map((col, idx) => (
              <th
                key={idx}
                style={{ textAlign: col.align || 'left', width: col.width }}
                className={col.className}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIdx) => {
            const key = keyExtractor ? keyExtractor(row, rowIdx) : rowIdx;
            const isSelected = selectedKey !== undefined && selectedKey === key;
            return (
              <tr
                key={key}
                onClick={() => onRowClick && onRowClick(row)}
                style={{
                  cursor: onRowClick ? 'pointer' : 'default',
                  backgroundColor: isSelected ? 'var(--bg-active)' : undefined,
                }}
              >
                {columns?.map((col, colIdx) => {
                  let content: React.ReactNode;
                  if (typeof col.accessor === 'function') {
                    content = col.accessor(row);
                  } else if (col.accessor) {
                    content = (row as any)[col.accessor];
                  } else {
                    content = null;
                  }

                  return (
                    <td
                      key={colIdx}
                      style={{ textAlign: col.align || 'left' }}
                      className={col.className}
                    >
                      {content}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const Thead: React.FC<React.HTMLAttributes<HTMLTableSectionElement>> = ({ children, ...props }) => (
  <thead {...props}>{children}</thead>
);

export const Tbody: React.FC<React.HTMLAttributes<HTMLTableSectionElement>> = ({ children, ...props }) => (
  <tbody {...props}>{children}</tbody>
);

export const Tr: React.FC<React.HTMLAttributes<HTMLTableRowElement>> = ({ children, ...props }) => (
  <tr {...props}>{children}</tr>
);

export const Th: React.FC<React.ThHTMLAttributes<HTMLTableCellElement>> = ({ children, ...props }) => (
  <th {...props}>{children}</th>
);

export const Td: React.FC<React.TdHTMLAttributes<HTMLTableCellElement>> = ({ children, ...props }) => (
  <td {...props}>{children}</td>
);
