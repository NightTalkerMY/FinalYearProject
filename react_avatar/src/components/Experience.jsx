import React, { useEffect, useRef, useState, Suspense } from 'react'; // Added Suspense
import { Environment, CameraControls } from "@react-three/drei";
import { Avatar } from "./Avatar";
import { Carousel } from "./carousel";
import { InspectionStage } from "./InspectionStage"; 
import { useOrchestrator } from "../hooks/useOrchestrator";
import { useSpring, animated } from '@react-spring/three';

export const Experience = ({ onAudioTrackReady }) => {
  const { 
    status, audioUrl, visemeUrl, gesture, 
    asins, triggerCarousel, resetState,
    updateId, triggerGoodbye 
  } = useOrchestrator();

  const controls = useRef();
  const [phase, setPhase] = useState(0);
  const [localIndex, setLocalIndex] = useState(0);
  
  // LOCKS
  const lastProcessedId = useRef(0);
  const lastGoodbyeId = useRef(-1); 

  // --- SMART RESET ---
  const prevFirstAsin = useRef(null);
  useEffect(() => {
    const currentFirstAsin = asins?.[0] || null;
    if (phase === 0 || currentFirstAsin !== prevFirstAsin.current) {
        setLocalIndex(0);
        prevFirstAsin.current = currentFirstAsin;
    }
  }, [asins, phase]);  

  // --- GESTURE LOGIC ---
  useEffect(() => {
    if (updateId > lastProcessedId.current) {
      console.log(`⚡ Phase ${phase} | Gesture: ${gesture} | ID: ${updateId}`);

      if (phase === 3) { // CAROUSEL
          const total = asins.length || 0;
          if (gesture === "swipe_left" || gesture === "left") setLocalIndex(p => (p > 0 ? p - 1 : 0));
          else if (gesture === "swipe_right" || gesture === "right") setLocalIndex(p => (p < total - 1 ? p + 1 : p));
          else if (gesture === "grab") setPhase(7); 
          else if (gesture === "expand") {
             if (lastGoodbyeId.current !== updateId) {
                 lastGoodbyeId.current = updateId;
                 triggerGoodbye(); 
                 setPhase(4);
             }
          }
      }
      else if (phase === 7) { // INSPECTION
          if (gesture === "expand") setPhase(3); 
      }
      
      lastProcessedId.current = updateId;
    }
  }, [updateId, gesture, phase, triggerGoodbye, asins]);

  // --- MOVEMENT ---
  const { avatarX } = useSpring({
    to: { 
        avatarX: (() => {
            if (phase === 0 || phase === 6) return 0;       
            if (phase === 5) return 0;
            if (phase === 1) return -3.0; 
            return -1.2; 
        })()
    },
    config: { mass: 2, tension: 60, friction: 40 }
  });

  const handleSpeechEnd = () => {
    if (triggerCarousel && phase === 0) setPhase(1);
    else resetState();
  };

  // --- TIMERS ---
  useEffect(() => {
    if (phase === 1) setTimeout(() => setPhase(2), 3000);
    if (phase === 2) setTimeout(() => setPhase(3), 1000);
    if (phase === 4) setTimeout(() => setPhase(5), 2000);
    if (phase === 5) setTimeout(() => setPhase(6), 1800);
    if (phase === 6) setTimeout(() => setPhase(0), 100);
  }, [phase]);

  // --- CAMERA ---
  useEffect(() => {
    if (!controls.current) return;
    if (phase === 0 || phase === 1 || phase === 5 || phase === 6) {
       controls.current.setLookAt(0, 1.5, 3.5, 0, 1.6, 0, true);
    }
    // Carousel View
    else if (phase === 2 || phase === 3 || phase === 4) {
       controls.current.setLookAt(0, 1.8, 6.0,  3.0, 2.0, 0, true);
    }
    // Inspection View
    else if (phase === 7) {
       controls.current.setLookAt(0, 1.5, 3.2,  3.5, 2.0, 0, true);
    }
  }, [phase]);

  const getAvatarState = () => {
    if (phase === 1) return "WALK_OFF";
    if (phase === 5) return "WALK_ON";
    if (phase === 0 || phase === 6) return "IDLE";
    return "WAITING"; 
  };

  const isAvatarVisible = (phase === 0 || phase === 1 || phase === 5 || phase === 6);

  const total = asins.length || 1;
  const safeIndex = Math.min(Math.max(localIndex, 0), total - 1); 
  const currentAsin = asins[safeIndex];

  return (
    <>
      <CameraControls ref={controls} />
      
      {/* AVATAR: Kept outside the product suspense boundary so it never blinks */}
      <animated.group position-x={avatarX}>
        <Avatar 
          scale={1.0} 
          onAudioTrackReady={onAudioTrackReady}
          status={status}
          audioUrl={audioUrl}
          visemeUrl={visemeUrl}
          gesture={gesture}
          sequenceState={getAvatarState()} 
          visible={isAvatarVisible}
          onSpeechEnded={handleSpeechEnd} 
        />
      </animated.group>

      {/* PRODUCTS: Wrapped in their own Suspense. 
          If a texture fails here, only this part might flicker/fallback, 
          but the avatar will remain solid. */}
      <Suspense fallback={null}>
          {(phase === 3 || phase === 4) && (
            <Carousel 
               asins={asins} 
               selectedIndex={safeIndex} 
               position={[3.0, 1.3, 0]} 
               isExiting={phase === 4} 
            />
          )}

          {currentAsin && (
            <InspectionStage 
               asin={currentAsin}
               visible={phase === 7}
               gesture={gesture}    
               updateId={updateId} 
            />
          )}
      </Suspense>
      
      <Environment preset="sunset" />
    </>
  );
};