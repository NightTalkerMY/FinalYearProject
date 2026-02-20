import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Canvas } from "@react-three/fiber";
import { Experience } from "./components/Experience";

const MEDIAMTX_URL = "http://localhost:8889/avatar/whip";

function App() {
  const canvasRef = useRef();
  const audioTrackRef = useRef(null);
  const pcRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const handleAudioTrack = useCallback((track) => {
    audioTrackRef.current = track;
  }, []);

  const startStreaming = async () => {
    if (isStreaming || !canvasRef.current) return;

    try {
      const videoStream = canvasRef.current.captureStream(60);
      const tracks = videoStream.getTracks();

      console.log("Detected tracks:", tracks.length); 
      if (tracks.length === 0) {
        console.error("No tracks found! Retrying in 1s...");
        setTimeout(startStreaming, 1000);
        return;
      }

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
      });

      pc.addTrack(tracks[0], videoStream);

      if (audioTrackRef.current) {
        pc.addTrack(audioTrackRef.current, videoStream);
      }

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const response = await fetch(MEDIAMTX_URL, {
        method: 'POST',
        body: pc.localDescription.sdp,
        headers: { 'Content-Type': 'application/sdp' }
      });

      const answerSdp = await response.text();
      await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });

      setIsStreaming(true);
    } catch (err) {
      console.error("Stream initialization failed:", err);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      startStreaming();
    }, 2000); 
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const heartbeat = setInterval(() => {
      if (pcRef.current && isStreaming) {
        pcRef.current.getStats(); 
      }
    }, 1000);
    return () => clearInterval(heartbeat);
  }, [isStreaming]);

  return (
    <div style={{ width: "100vw", height: "100vh", backgroundColor: "#000" }}>
      <Canvas 
        ref={canvasRef} 
        shadows 
        // --- CAMERA FIX ---
        // position: [x, y, z] -> [0, 1.5, 3.5]
        // y=1.5: Eye level
        // z=3.5: 3.5 meters away (Close enough to see face, far enough for body)
        // fov=30: Narrower lens for a "Portrait" look (less fisheye distortion)
        camera={{ position: [0, 1.65, 3.0], fov: 30 }}
        gl={{ preserveDrawingBuffer: true, antialias: true }} 
      >
        <color attach="background" args={["#000000"]} />
        <Experience onAudioTrackReady={handleAudioTrack} />
      </Canvas>
    </div>
  );
}

export default App;