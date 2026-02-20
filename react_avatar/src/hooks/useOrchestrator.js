import { useState, useEffect, useRef, useCallback } from 'react'; // Added useCallback

const ORCHESTRATOR_URL = "http://localhost:5000";

export function useOrchestrator() {
  const [state, setState] = useState({
    status: "IDLE",       
    audioUrl: null,
    visemeUrl: null,
    triggerCarousel: false,
    asins: [],
    gesture: "talk",
    updateId: 0
  });

  const lastUpdateIdRef = useRef(0);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${ORCHESTRATOR_URL}/poll_state`);
        const data = await res.json();

        if (data.last_update_id !== lastUpdateIdRef.current) {
          lastUpdateIdRef.current = data.last_update_id;
          
          setState({
            status: data.status,
            audioUrl: data.audio_url,
            visemeUrl: data.viseme_url,
            triggerCarousel: data.trigger_carousel,
            asins: data.asins || [],
            gesture: data.gesture || "talk",
            updateId: data.last_update_id 
          });
        }
      } catch (err) {
        // console.warn("Orchestrator offline?", err);
      }
    };

    const interval = setInterval(poll, 50); 
    return () => clearInterval(interval);
  }, []);

  const resetState = async () => {
    try {
      await fetch(`${ORCHESTRATOR_URL}/reset_state`, { method: "POST" });
      setState(prev => ({ ...prev, status: "IDLE" }));
    } catch (e) { console.error(e); }
  };

  // FIX: Wrapped in useCallback to prevent re-creation
  const triggerGoodbye = useCallback(async () => {
    try {
        await fetch(`${ORCHESTRATOR_URL}/generate_goodbye`, { method: "POST" });
    } catch (e) { console.error(e); }
  }, []);

  return { ...state, resetState, triggerGoodbye };
}