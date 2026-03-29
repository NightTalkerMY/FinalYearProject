import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { ProductCard } from './ProductCard';
import { useSpring, animated } from '@react-spring/three';
import { Text } from '@react-three/drei';

export function Carousel({ asins, selectedIndex = 0, position, isExiting }) {
  const { slideY } = useSpring({
    to: { slideY: isExiting ? -5 : position[1] }, 
    from: { slideY: -5 },
    config: { mass: 1, tension: 60, friction: 15 }
  });

  if (!asins || asins.length === 0) return null;

  return (
    <animated.group position-x={position[0]} position-y={slideY} position-z={position[2]}>
      
      {/* --- ACTIVE ITEM HIGHLIGHT --- */}
      <group position={[0, -0.4, 0]}>
         <mesh rotation={[-Math.PI / 2, 0, 0]}>
            {/* FIX: Shrunk the ring from [0.7, 0.9] to [0.45, 0.55] */}
            <ringGeometry args={[0.45, 0.55, 32]} />
            <meshStandardMaterial color="#00ffcc" emissive="#00ffcc" emissiveIntensity={3} />
         </mesh>
         <pointLight position={[0, 2.0, 1.5]} intensity={3} color="#ffffff" distance={5} />
      </group>

      {/* --- ITEMS --- */}
      {asins.map((asin, i) => {
        const offset = i - selectedIndex;
        return (
          <SliderItem 
             key={asin}
             asin={asin}
             offset={offset}
             isSelected={i === selectedIndex}
          />
        );
      })}
    </animated.group>
  );
}

function SliderItem({ asin, offset, isSelected }) {
    const targetX = offset * 0.85; 
    const targetZ = Math.abs(offset) * -1.5; 
    
    // --- NEW LOGIC: Only show selected item and immediate neighbors ---
    const isNeighbor = Math.abs(offset) === 1;
    const scaleTarget = isSelected ? 1.0 : (isNeighbor ? 0.45 : 0.0); 
    
    const rotTarget = isSelected ? 0.2 : offset * -0.4; 

    const { x, z, s, rot } = useSpring({
        x: targetX,
        z: targetZ,
        s: scaleTarget,
        rot: rotTarget, 
        config: { mass: 1, tension: 100, friction: 20 }
    });

    return (
        <animated.group position-x={x} position-z={z} scale={s} rotation-y={rot}>
            <ProductCard asin={asin} isSelected={isSelected} />
        </animated.group>
    );
}