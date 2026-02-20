import React, { useRef, useState } from 'react';
import { useGLTF, Float, Resize, Center } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';

export function ProductCard({ asin, isSelected }) {
  const group = useRef();
  
  const path = `/products/${asin}/model.gltf`; 
  const { scene } = useGLTF(path);
  const [hovered, setHover] = useState(false);

  useFrame((state, delta) => {
    if (group.current) {
      if (isSelected) {
        group.current.rotation.y += delta * 0.5;
      } else {
        group.current.rotation.y = 0;
      }
    }
  });

  return (
    <group ref={group}>
      <Float speed={2} rotationIntensity={0.2} floatIntensity={0.2}>
        <Resize scale={1.5}>
          {/* FIX: Removed 'top' prop. Now it centers perfectly on the axis. */}
          <Center>
             <primitive 
                object={scene} 
                onPointerOver={() => setHover(true)}
                onPointerOut={() => setHover(false)}
             />
          </Center>
        </Resize>
      </Float>
    </group>
  );
}