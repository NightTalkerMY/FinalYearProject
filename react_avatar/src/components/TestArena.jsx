import React, { useEffect, useMemo, useRef } from 'react';
import { Canvas, useGraph } from '@react-three/fiber';
import { useGLTF, useFBX, useAnimations, OrbitControls, Environment } from '@react-three/drei';
import { SkeletonUtils } from 'three-stdlib';

function TestAvatar() {
  const group = useRef();
  const { scene } = useGLTF('/models/wyy.glb');
  const clone = useMemo(() => SkeletonUtils.clone(scene), [scene]);
  const { nodes, materials } = useGraph(clone);

  const { animations: idleData } = useFBX("animations/boy/Idle.fbx");

const animations = useMemo(() => {
    const clip = idleData[0].clone(); 
    clip.name = "idle";
    
    // 1. Remove Scale Tracks (Keep this!)
    clip.tracks = clip.tracks.filter((track) => !track.name.endsWith(".scale"));

    // 2. Fix Bone Names AND Scale Positions
    clip.tracks.forEach((track) => {
      // Fix Name
      track.name = track.name.replace("mixamorig", "");

      // Fix Position (The Magic Fix)
      // If this track controls position, divide the values by 100
      if (track.name.endsWith(".position")) {
        for (let i = 0; i < track.values.length; i++) {
          track.values[i] *= 0.01; // Convert cm to meters!
        }
      }
    });

    return [clip];
  }, [idleData]);

  const { actions } = useAnimations(animations, group);

  useEffect(() => {
    if (actions.idle) {
      actions.idle.reset().fadeIn(0.5).play();
    }
  }, [actions]);

  return (
    // FIX: Manually force scale here. 
    // Try [1, 1, 1] first. If tiny, try [100, 100, 100]. If huge, try [0.01, 0.01, 0.01].
    // Since it looks like a speck, it's likely needing a 100x boost or the camera is too far.
    <group ref={group} dispose={null} scale={[1, 1, 1]}> 
      <primitive object={clone} />

      {/* --- BODY PARTS --- */}
      {nodes.Body_Mesh && <skinnedMesh geometry={nodes.Body_Mesh.geometry} material={materials.Body} skeleton={nodes.Body_Mesh.skeleton} />}
      {nodes.avaturn_hair_0 && <skinnedMesh geometry={nodes.avaturn_hair_0.geometry} material={materials.avaturn_hair_0_material} skeleton={nodes.avaturn_hair_0.skeleton} />}
      {nodes.avaturn_hair_1 && <skinnedMesh geometry={nodes.avaturn_hair_1.geometry} material={materials.avaturn_hair_1_material} skeleton={nodes.avaturn_hair_1.skeleton} />}
      {nodes.avaturn_shoes_0 && <skinnedMesh geometry={nodes.avaturn_shoes_0.geometry} material={materials.avaturn_shoes_0_material} skeleton={nodes.avaturn_shoes_0.skeleton} />}
      {nodes.avaturn_look_0 && <skinnedMesh geometry={nodes.avaturn_look_0.geometry} material={materials.avaturn_look_0_material} skeleton={nodes.avaturn_look_0.skeleton} />}
      
      {/* --- FACE PARTS --- */}
      {nodes.Head_Mesh && (
        <skinnedMesh 
          geometry={nodes.Head_Mesh.geometry} 
          material={materials.Head} 
          skeleton={nodes.Head_Mesh.skeleton} 
          morphTargetDictionary={nodes.Head_Mesh.morphTargetDictionary}
          morphTargetInfluences={nodes.Head_Mesh.morphTargetInfluences}
        />
      )}
      
      {nodes.Teeth_Mesh && (
        <skinnedMesh 
          geometry={nodes.Teeth_Mesh.geometry} 
          material={materials.Teeth} 
          skeleton={nodes.Teeth_Mesh.skeleton} 
          morphTargetDictionary={nodes.Teeth_Mesh.morphTargetDictionary}
          morphTargetInfluences={nodes.Teeth_Mesh.morphTargetInfluences}
        />
      )}
      
      {nodes.Eye_Mesh && (
        <skinnedMesh 
          geometry={nodes.Eye_Mesh.geometry} 
          material={materials.Eyes} 
          skeleton={nodes.Eye_Mesh.skeleton} 
          morphTargetDictionary={nodes.Eye_Mesh.morphTargetDictionary}
          morphTargetInfluences={nodes.Eye_Mesh.morphTargetInfluences}
        />
      )}
    </group>
  );
}

export default function TestArena() {
  return (
    <div style={{ width: "100vw", height: "100vh", backgroundColor: "#333" }}>
      {/* FIX: Move camera closer to see if it's just a camera issue */}
      <Canvas camera={{ position: [0, 1.5, 2] }}> 
        <OrbitControls target={[0, 1.0, 0]} />
        <Environment preset="sunset" />
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <TestAvatar />
        {/* Helper to see where 0,0,0 is */}
        <gridHelper /> 
        <axesHelper args={[5]} />
      </Canvas>
    </div>
  );
}