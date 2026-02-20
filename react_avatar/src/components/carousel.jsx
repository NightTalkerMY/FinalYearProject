import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { ProductCard } from './ProductCard';
import { useSpring, animated } from '@react-spring/three';
import { Text } from '@react-three/drei';

export function Carousel({ asins, selectedIndex = 0, position, isExiting }) {
  const SPACING = 2.5; 

  const { slideY } = useSpring({
    // FIX: Changed 0.5 -> 1.5 to shift the whole carousel UP
    to: { slideY: isExiting ? -5 : 1.5 }, 
    from: { slideY: -5 },
    config: { mass: 1, tension: 60, friction: 15 }
  });

  if (!asins || asins.length === 0) return null;

  return (
    <animated.group position-x={position[0]} position-y={slideY} position-z={position[2]}>
      
      {/* --- ACTIVE ITEM HIGHLIGHT --- */}
      <group position={[0, -0.6, 0]}>
         <mesh rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.7, 0.9, 32]} />
            <meshStandardMaterial color="#00ffcc" emissive="#00ffcc" emissiveIntensity={3} />
         </mesh>
         <pointLight position={[0, 0.5, 0]} intensity={2} color="#00ffcc" distance={2} />
      </group>

      {/* --- ITEMS --- */}
      {asins.map((asin, i) => {
        const offset = i - selectedIndex;
        const targetX = offset * SPACING;
        const scale = i === selectedIndex ? 1.0 : 0.7;
        
        return (
          <SliderItem 
             key={asin}
             asin={asin}
             targetX={targetX}
             scale={scale}
             isSelected={i === selectedIndex}
          />
        );
      })}
    </animated.group>
  );
}

function SliderItem({ asin, targetX, scale, isSelected }) {
    const { x, s, rot } = useSpring({
        x: targetX,
        s: scale,
        rot: isSelected ? 0.2 : 0, 
        config: { mass: 1, tension: 100, friction: 20 }
    });

    return (
        <animated.group position-x={x} scale={s} rotation-y={rot}>
            <ProductCard asin={asin} isSelected={isSelected} />
        </animated.group>
    );
}