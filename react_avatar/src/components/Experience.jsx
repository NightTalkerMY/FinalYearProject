// import React, { useEffect, useRef, useState, Suspense } from 'react'; // Added Suspense
// import { Environment, CameraControls } from "@react-three/drei";
// import { Avatar } from "./Avatar";
// import { Carousel } from "./carousel";
// import { InspectionStage } from "./InspectionStage"; 
// import { useOrchestrator } from "../hooks/useOrchestrator";
// import { useSpring, animated } from '@react-spring/three';

// export const Experience = ({ onAudioTrackReady }) => {
//   const { 
//     status, audioUrl, visemeUrl, gesture, 
//     asins, triggerCarousel, resetState,
//     updateId, triggerGoodbye, setFocusAsin
//   } = useOrchestrator();

//   const controls = useRef();
//   const [phase, setPhase] = useState(0);
//   const [localIndex, setLocalIndex] = useState(0);
  
//   // LOCKS
//   const lastProcessedId = useRef(0);
//   const lastGoodbyeId = useRef(-1); 

//   // --- SMART RESET ---
//   const prevFirstAsin = useRef(null);
//   useEffect(() => {
//     const currentFirstAsin = asins?.[0] || null;
//     if (phase === 0 || currentFirstAsin !== prevFirstAsin.current) {
//         setLocalIndex(0);
//         prevFirstAsin.current = currentFirstAsin;
//     }
//   }, [asins, phase]);  

//   // --- GESTURE LOGIC ---
//   useEffect(() => {
//     if (updateId > lastProcessedId.current) {
//       console.log(`⚡ Phase ${phase} | Gesture: ${gesture} | ID: ${updateId}`);

//       if (phase === 3) { // CAROUSEL
//           const total = asins.length || 0;
//           if (gesture === "swipe_left" || gesture === "left") setLocalIndex(p => (p > 0 ? p - 1 : 0));
//           else if (gesture === "swipe_right" || gesture === "right") setLocalIndex(p => (p < total - 1 ? p + 1 : p));
//           else if (gesture === "grab") setPhase(7); 
//           else if (gesture === "expand") {
//              if (lastGoodbyeId.current !== updateId) {
//                  lastGoodbyeId.current = updateId;
//                  triggerGoodbye(); 
//                  setPhase(4);
//              }
//           }
//       }
//       else if (phase === 7) { // INSPECTION
//           if (gesture === "expand") setPhase(3); 
//       }
      
//       lastProcessedId.current = updateId;
//     }
//   }, [updateId, gesture, phase, triggerGoodbye, asins]);

//   // --- MOVEMENT ---
//   const { avatarX } = useSpring({
//     to: { 
//         avatarX: (() => {
//             if (phase === 0 || phase === 6) return 0;       
//             if (phase === 5) return 0;
//             if (phase === 1) return -3.0; 
//             return -1.2; 
//         })()
//     },
//     config: { mass: 2, tension: 60, friction: 40 }
//   });

//   const handleSpeechEnd = () => {
//     if (triggerCarousel && phase === 0) setPhase(1);
//     else resetState();
//   };

//   // --- TIMERS ---
//   useEffect(() => {
//     if (phase === 1) setTimeout(() => setPhase(2), 3000);
//     if (phase === 2) setTimeout(() => setPhase(3), 1000);
//     if (phase === 4) setTimeout(() => setPhase(5), 2000);
//     if (phase === 5) setTimeout(() => setPhase(6), 1800);
//     if (phase === 6) setTimeout(() => setPhase(0), 100);
//   }, [phase]);

//   // --- CAMERA ---
//   useEffect(() => {
//     if (!controls.current) return;
//     if (phase === 0 || phase === 1 || phase === 5 || phase === 6) {
//        controls.current.setLookAt(0, 1.5, 3.5, 0, 1.6, 0, true);
//     }
//     // Carousel View
//     else if (phase === 2 || phase === 3 || phase === 4) {
//        controls.current.setLookAt(0, 1.8, 6.0,  3.0, 2.0, 0, true);
//     }
//     // Inspection View
//     else if (phase === 7) {
//        controls.current.setLookAt(0, 1.5, 3.2,  3.5, 2.0, 0, true);
//     }
//   }, [phase]);

  
//   const total = asins.length || 1;
//   const safeIndex = Math.min(Math.max(localIndex, 0), total - 1); 
//   const currentAsin = asins[safeIndex];

//   // --- NEW: SYNC FOCUS WITH BACKEND ---
//   useEffect(() => {
//     // Only send focus updates if we are in the Carousel phases
//     if ((phase === 3 || phase === 4 || phase === 7) && currentAsin) {
//        setFocusAsin(currentAsin);
//     }
//   }, [currentAsin, phase, setFocusAsin]);

//   const getAvatarState = () => {
//     if (phase === 1) return "WALK_OFF";
//     if (phase === 5) return "WALK_ON";
//     if (phase === 0 || phase === 6) return "IDLE";
//     return "WAITING"; 
//   };

//   const isAvatarVisible = (phase === 0 || phase === 1 || phase === 5 || phase === 6);

//   return (
//     <>
//       <CameraControls ref={controls} />
      
//       {/* AVATAR: Kept outside the product suspense boundary so it never blinks */}
//       <animated.group position-x={avatarX}>
//         <Avatar 
//           scale={1.0} 
//           onAudioTrackReady={onAudioTrackReady}
//           status={status}
//           audioUrl={audioUrl}
//           visemeUrl={visemeUrl}
//           gesture={gesture}
//           sequenceState={getAvatarState()} 
//           visible={isAvatarVisible}
//           onSpeechEnded={handleSpeechEnd} 
//         />
//       </animated.group>

//       {/* PRODUCTS: Wrapped in their own Suspense. 
//           If a texture fails here, only this part might flicker/fallback, 
//           but the avatar will remain solid. */}
//       <Suspense fallback={null}>
//           {(phase === 3 || phase === 4) && (
//             <Carousel 
//                asins={asins} 
//                selectedIndex={safeIndex} 
//                position={[3.0, 1.3, 0]} 
//                isExiting={phase === 4} 
//             />
//           )}

//           {currentAsin && (
//             <InspectionStage 
//                asin={currentAsin}
//                visible={phase === 7}
//                gesture={gesture}    
//                updateId={updateId} 
//             />
//           )}
//       </Suspense>
      
//       <Environment preset="sunset" />
//     </>
//   );
// };

// import React, { useEffect, useRef, useState, Suspense } from 'react';
// import { Environment, CameraControls } from "@react-three/drei";
// import { Avatar } from "./Avatar";
// import { Carousel } from "./carousel";
// import { InspectionStage } from "./InspectionStage"; 
// import { useOrchestrator } from "../hooks/useOrchestrator";
// import { useSpring, animated } from '@react-spring/three';

// export const Experience = ({ onAudioTrackReady }) => {
//   const { 
//     status, audioUrl, visemeUrl, gesture, 
//     asins, triggerCarousel, resetState,
//     updateId, triggerGoodbye, setFocusAsin
//   } = useOrchestrator();

//   const controls = useRef();
//   const [phase, setPhase] = useState(0);
//   const [localIndex, setLocalIndex] = useState(0);
  
//   // LOCKS
//   const lastProcessedId = useRef(0);
//   const lastGoodbyeId = useRef(-1); 

//   // --- SMART RESET ---
//   const prevFirstAsin = useRef(null);
//   useEffect(() => {
//     const currentFirstAsin = asins?.[0] || null;
//     if (phase === 0 || currentFirstAsin !== prevFirstAsin.current) {
//         setLocalIndex(0);
//         prevFirstAsin.current = currentFirstAsin;
//     }
//   }, [asins, phase]);  

//   // --- GESTURE LOGIC ---
//   useEffect(() => {
//     if (updateId > lastProcessedId.current) {
//       console.log(`⚡ Phase ${phase} | Gesture: ${gesture} | ID: ${updateId}`);

//       if (phase === 3) { // CAROUSEL MODE
//           const total = asins.length || 0;
//           if (gesture === "swipe_left" || gesture === "left") setLocalIndex(p => (p > 0 ? p - 1 : 0));
//           else if (gesture === "swipe_right" || gesture === "right") setLocalIndex(p => (p < total - 1 ? p + 1 : p));
//           else if (gesture === "grab") setPhase(7); // Trigger Avatar to walk off
//           else if (gesture === "expand") {
//              if (lastGoodbyeId.current !== updateId) {
//                  lastGoodbyeId.current = updateId;
//                  triggerGoodbye(); 
//                  setPhase(4); // Trigger Carousel to exit
//              }
//           }
//       }
//       else if (phase === 8) { // INSPECTION MODE
//           if (gesture === "expand") setPhase(9); // Trigger Avatar to walk back
//       }
      
//       lastProcessedId.current = updateId;
//     }
//   }, [updateId, gesture, phase, triggerGoodbye, asins]);

//   // --- MOVEMENT MATH (Left Zone / Right Zone) ---
//   const { avatarX, avatarRotY } = useSpring({
//     to: { 
//         avatarX: (() => {
//             if (phase === 0 || phase === 5 || phase === 6) return 0;      
//             if (phase === 1 || phase === 2 || phase === 3 || phase === 4 || phase === 9) return -1.2; 
//             if (phase === 7 || phase === 8) return -5.0; 
//             return 0; 
//         })(),
//         // NEW: Rotate him slightly to face the center when he stands on the left!
//         avatarRotY: (() => {
//             if (phase === 0 || phase === 5 || phase === 6) return 0; // Face straight ahead
//             if (phase === 1 || phase === 2 || phase === 3 || phase === 4 || phase === 9) return 0.25; // Turn slightly towards center
//             return 0; 
//         })()
//     },
//     config: { mass: 2, tension: 60, friction: 40 }
//   });

//   const handleSpeechEnd = () => {
//     if (triggerCarousel && phase === 0) setPhase(1);
//     else resetState();
//   };

//   // --- THE NEW TIMERS ---
//   useEffect(() => {
//     if (phase === 1) setTimeout(() => setPhase(2), 2000); // Walk Left -> Settle
//     if (phase === 2) setTimeout(() => setPhase(3), 500);  // Settle -> Carousel Up
//     if (phase === 4) setTimeout(() => setPhase(5), 1500); // Carousel Down -> Walk Right
//     if (phase === 5) setTimeout(() => setPhase(6), 2000); // Walk Right -> Settle
//     if (phase === 6) setTimeout(() => setPhase(0), 100);  // Return to Idle
    
//     // Inspection Transitions
//     if (phase === 7) setTimeout(() => setPhase(8), 2000); // Walk Off -> Inspection Active
//     if (phase === 9) setTimeout(() => setPhase(3), 2000); // Walk On -> Carousel Active
//   }, [phase]);

//   // --- CAMERA ---
//   useEffect(() => {
//     if (!controls.current) return;
    
//     // Mode 1: Avatar Center
//     if (phase === 0 || phase === 5 || phase === 6) {
//        controls.current.setLookAt(0, 1.5, 3.5,  0, 1.5, 0, true);
//     }
//     // Mode 2: Avatar & Carousel
//     else if (phase === 2 || phase === 3 || phase === 4 || phase === 9) {
//        // <--- Changed Z from 6.5 to 4.5, and pan slightly left to balance
//        controls.current.setLookAt(-0.4, 1.4, 5.2,  -0.4, 1.0, 0, true);
//     }
//     // Mode 3: Inspection
//     else if (phase === 7 || phase === 8) {
//        controls.current.setLookAt(0.3, 1.5, 3.5,  0.3, 1.5, 0, true);
//     }
//   }, [phase]);

  
//   const total = asins.length || 1;
//   const safeIndex = Math.min(Math.max(localIndex, 0), total - 1); 
//   const currentAsin = asins[safeIndex];

//   // --- SYNC FOCUS WITH BACKEND ---
//   useEffect(() => {
//     // Ping backend when swiping in Carousel or rotating in Inspection
//     if ((phase === 3 || phase === 4 || phase === 8) && currentAsin) {
//        setFocusAsin(currentAsin);
//     }
//   }, [currentAsin, phase, setFocusAsin]);

//   const getAvatarState = () => {
//     if (phase === 1 || phase === 7) return "WALK_OFF"; // Triggers walkLeft
//     if (phase === 5 || phase === 9) return "WALK_ON";  // Triggers walkRight
//     return "IDLE"; // Force Idle for all other phases to allow speaking
//   };

//   // Keep avatar rendered unless completely off-screen in phase 8
//   const isAvatarVisible = phase !== 8;

//   return (
//     <>
//       <CameraControls ref={controls} />
      
//       {/* AVATAR */}
//       <animated.group position-x={avatarX} rotation-y={avatarRotY}>
//         <Avatar 
//           scale={1.0} 
//           onAudioTrackReady={onAudioTrackReady}
//           status={status}
//           audioUrl={audioUrl}
//           visemeUrl={visemeUrl}
//           gesture={gesture}
//           sequenceState={getAvatarState()} 
//           visible={isAvatarVisible}
//           onSpeechEnded={handleSpeechEnd} 
//         />
//       </animated.group>

//       {/* PRODUCTS */}
//       <Suspense fallback={null}>
//           {(phase === 2 || phase === 3 || phase === 4 || phase === 7 || phase === 9) && (
//             <Carousel 
//                asins={asins} 
//                selectedIndex={safeIndex} 
//                position={[0.3, 0.8, 0]} // <--- Moved X from 1.8 down to 0.8
//                isExiting={phase === 4 || phase === 7}
//             />
//           )}

//           {currentAsin && (
//             <InspectionStage 
//                asin={currentAsin}
//                visible={phase === 7 || phase === 8}
//                gesture={gesture}    
//                updateId={updateId} 
//             />
//           )}
//       </Suspense>
      
//       <Environment preset="sunset" />
//     </>
//   );
// };


// Last full response try
import React, { useEffect, useRef, useState, Suspense } from 'react';
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
    updateId, triggerGoodbye, setFocusAsin
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

      if (phase === 3) { // CAROUSEL MODE
          const total = asins.length || 0;
          if (gesture === "swipe_left" || gesture === "left") setLocalIndex(p => (p > 0 ? p - 1 : 0));
          else if (gesture === "swipe_right" || gesture === "right") setLocalIndex(p => (p < total - 1 ? p + 1 : p));
          else if (gesture === "grab") setPhase(7); // Trigger Avatar to walk off
          else if (gesture === "expand") {
             if (lastGoodbyeId.current !== updateId) {
                 lastGoodbyeId.current = updateId;
                 triggerGoodbye(); 
                 setPhase(4); // Trigger Carousel to exit
             }
          }
      }
      else if (phase === 8) { // INSPECTION MODE
          if (gesture === "expand") setPhase(9); // Trigger Avatar to walk back
      }
      
      lastProcessedId.current = updateId;
    }
  }, [updateId, gesture, phase, triggerGoodbye, asins]);

  // --- MOVEMENT MATH (Left Zone / Right Zone) ---
  const { avatarX, avatarRotY } = useSpring({
    to: { 
        avatarX: (() => {
            // FIX: Center targets (includes Phase 10 and Phase 5 returning)
            if (phase === 1 || phase === 2 || phase === 3 || phase === 4 || phase === 9) return -1.2; 
            if (phase === 7 || phase === 8) return -5.0; 
            return 0; 
        })(),
        avatarRotY: (() => {
            // Only turn towards the carousel when fully settled on the left
            if (phase === 2 || phase === 3 || phase === 4 || phase === 9) return 0.25; 
            return 0; 
        })()
    },
    config: { mass: 2, tension: 60, friction: 40 }
  });

  const handleSpeechEnd = () => {
    // FIX: Instead of jumping to Phase 1 (Walking), we jump to Phase 10 (Camera Zoom Out First)
    if (triggerCarousel && phase === 0) setPhase(10); 
    else resetState();
  };

  // --- THE NEW TIMERS ---
  useEffect(() => {
    // ENTRANCE SEQUENCE
    if (phase === 10) setTimeout(() => setPhase(1), 1000); // Zoom Out -> Wait 1s -> Start Walk
    if (phase === 1) setTimeout(() => setPhase(2), 2000);  // Walk Left -> Settle
    if (phase === 2) setTimeout(() => setPhase(3), 500);   // Settle -> Carousel Up
    
    // EXIT SEQUENCE
    if (phase === 4) setTimeout(() => setPhase(5), 1000);  // Carousel Down -> Wait 1s -> Start Walk
    if (phase === 5) setTimeout(() => setPhase(6), 2000);  // Walk Right to Center -> Settle
    if (phase === 6) setTimeout(() => setPhase(0), 1000);  // Zoom In -> Wait 1s -> Idle
    
    // INSPECTION TRANSITIONS
    if (phase === 7) setTimeout(() => setPhase(8), 2000); // Walk Off -> Inspection Active
    if (phase === 9) setTimeout(() => setPhase(3), 2000); // Walk On -> Carousel Active
  }, [phase]);

  // --- CAMERA ---
  useEffect(() => {
    if (!controls.current) return;
    
    // Mode 1: Avatar Center (Tight Shot)
    // FIX: Only tight when fully idle (0) or during the final zoom-in (6)
    if (phase === 0 || phase === 6) {
       controls.current.setLookAt(0, 1.5, 3.5,  0, 1.5, 0, true);
    }
    // Mode 2: Avatar & Carousel (Wide Shot)
    // FIX: Include 10 (Zoom out) and 5 (Walk Right) to hold the wide shot until he is done walking
    else if (phase === 10 || phase === 1 || phase === 2 || phase === 3 || phase === 4 || phase === 5 || phase === 9) {
       controls.current.setLookAt(-0.4, 1.4, 5.2,  -0.4, 1.0, 0, true);
    }
    // Mode 3: Inspection
    else if (phase === 7 || phase === 8) {
       controls.current.setLookAt(0.3, 1.5, 3.5,  0.3, 1.5, 0, true);
    }
  }, [phase]);

  
  const total = asins.length || 1;
  const safeIndex = Math.min(Math.max(localIndex, 0), total - 1); 
  const currentAsin = asins[safeIndex];

  // --- SYNC FOCUS WITH BACKEND ---
  useEffect(() => {
    if ((phase === 3 || phase === 4 || phase === 8) && currentAsin) {
       setFocusAsin(currentAsin);
    }
  }, [currentAsin, phase, setFocusAsin]);

  const getAvatarState = () => {
    if (phase === 1 || phase === 7) return "WALK_OFF"; 
    if (phase === 5 || phase === 9) return "WALK_ON";  
    return "IDLE"; 
  };

  const isAvatarVisible = phase !== 8;

  return (
    <>
      <CameraControls ref={controls} />
      
      {/* AVATAR */}
      <animated.group position-x={avatarX} rotation-y={avatarRotY}>
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

      {/* PRODUCTS */}
      <Suspense fallback={null}>
          {(phase === 2 || phase === 3 || phase === 4 || phase === 7 || phase === 9) && (
            <Carousel 
               asins={asins} 
               selectedIndex={safeIndex} 
               position={[0.1, 0.8, 0]} 
               isExiting={phase === 4 || phase === 7}
            />
          )}

          {currentAsin && (
            <InspectionStage 
               asin={currentAsin}
               visible={phase === 7 || phase === 8}
               gesture={gesture}    
               updateId={updateId} 
            />
          )}
      </Suspense>
      
      <Environment preset="sunset" />
    </>
  );
};