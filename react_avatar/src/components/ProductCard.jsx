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
        {/* FIX: Changed scale from 1.5 to 0.5 to make it proportional to the avatar */}
        <Resize scale={0.85}> 
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