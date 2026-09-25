import { useState, useEffect, useRef } from 'react';
import { AnalysisEvent } from '../types';

export function useAnalysisWebSocket(analysisId: string | null) {
  const [latestEvent, setLatestEvent] = useState<AnalysisEvent | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!analysisId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/ws/analysis/${analysisId}`;

    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.stage) {
          setLatestEvent(parsed);
        }
      } catch (err) {
        // ignore ping/pong non-json
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
    };

    socket.onerror = () => {
      setIsConnected(false);
    };

    return () => {
      socket.close();
    };
  }, [analysisId]);

  return { latestEvent, isConnected };
}
