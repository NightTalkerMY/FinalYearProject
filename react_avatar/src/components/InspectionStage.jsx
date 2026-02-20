import React, { useRef, useState, useEffect } from 'react';
import { useGLTF, SpotLight } from '@react-three/drei';
import { useSpring, animated } from '@react-spring/three';

export function InspectionStage({ asin, visible, gesture, updateId }) {
  const lastProcessedId = useRef(0);
  const [targetRot, setTargetRot] = useState([0, 0]);

  const path = `/products/${asin}/model.gltf`; 
  const { scene } = useGLTF(path);

  // --- FIX: RESET ROTATION ON EXIT ---
  // This ensures every time you enter inspection, it starts fresh.
  useEffect(() => {
    if (!visible) {
        setTargetRot([0, 0]);
    }
  }, [visible]);

  // --- GESTURE LISTENER ---
  useEffect(() => {
    if (!visible) return;

    if (updateId > lastProcessedId.current) {
      const halfPi = Math.PI / 2;

      if (gesture === "swipe_left" || gesture === "left") setTargetRot(p => [p[0], p[1] - halfPi]);
      else if (gesture === "swipe_right" || gesture === "right") setTargetRot(p => [p[0], p[1] + halfPi]);
      else if (gesture === "swipe_up" || gesture === "up") setTargetRot(p => [p[0] - halfPi, p[1]]);
      else if (gesture === "swipe_down" || gesture === "down") setTargetRot(p => [p[0] + halfPi, p[1]]);

      lastProcessedId.current = updateId;
    }
  }, [updateId, gesture, visible]);

  // --- ANIMATION ---
  const { rotation, scale } = useSpring({
    rotation: [targetRot[0], targetRot[1], 0],
    scale: visible ? 4.8 : 0, 
    config: { mass: 1, tension: 120, friction: 26 }
  });

  if (!visible && scale.get() < 0.01) return null;

  return (
    <group position={[3.5, 1.8, 0]}> 
      
      {/* HERO LIGHTING */}
      {visible && (
        <>
          <SpotLight
             position={[2, 5, 5]} 
             angle={0.5} 
             penumbra={1} 
             intensity={3} 
             castShadow 
             color="white"
          />
          <pointLight position={[-3, 0, 3]} intensity={2} color="#ffffff" />
          <pointLight position={[0, -3, 0]} intensity={1} color="#aaaaff" />
        </>
      )}

      {/* THE PRODUCT */}
      <animated.group rotation={rotation} scale={scale}>
         <primitive object={scene} />
      </animated.group>

    </group>
  );
}