import React, { useEffect, useRef, useState, useMemo } from 'react'
import { useFrame, useGraph } from '@react-three/fiber'
import { useAnimations, useFBX, useGLTF } from '@react-three/drei'
import { SkeletonUtils } from 'three-stdlib'
import * as THREE from 'three'

const correspondings = { A: "viseme_PP", B: "viseme_kk", C: "viseme_I", D: "viseme_AA", E: "viseme_O", F: "viseme_U", H: "viseme_TH", X: "viseme_PP" }
const jawOpenByViseme = { viseme_sil: 0.0, viseme_PP: 0.0, viseme_FF: 0.1, viseme_TH: 0.15, viseme_DD: 0.1, viseme_kk: 0.2, viseme_CH: 0.25, viseme_SS: 0.1, viseme_nn: 0.05, viseme_RR: 0.1, viseme_aa: 0.45, viseme_E: 0.35, viseme_I: 0.3, viseme_O: 0.4, viseme_U: 0.4 };

export function Avatar(props) {
  const { 
    status, audioUrl, visemeUrl, gesture, resetState, 
    onAudioTrackReady, onSpeechEnded, 
    sequenceState, visible = true 
  } = props;

  const group = useRef()
  const { scene } = useGLTF('/models/wyy.glb')
  const clone = useMemo(() => SkeletonUtils.clone(scene), [scene])
  const { nodes, materials } = useGraph(clone)

  const { animations: idleData } = useFBX("animations/boy/Idle.fbx")
  const { animations: talkData } = useFBX("animations/boy/Talking_onefingerexplain(default).fbx")
  const { animations: transData } = useFBX("animations/boy/Talking_onepalmup(gesture-in-out).fbx")
  const { animations: confusedData } = useFBX("animations/boy/Talking_doublepalmup(na).fbx")
  const { animations: walkRightData } = useFBX("animations/boy/Left Strafe Walk.fbx") 
  const { animations: walkLeftData } = useFBX("animations/boy/Right Strafe Walk.fbx")

  const animations = useMemo(() => {
    const cleanAnimation = (anim, newName) => {
      const cloned = anim.clone();
      cloned.name = newName;
      cloned.tracks = cloned.tracks.filter((track) => !track.name.endsWith(".scale"));
      
      // FORCE IN-PLACE for Walks
      if (newName === "walkLeft" || newName === "walkRight") {
        cloned.tracks = cloned.tracks.filter((track) => {
            const isPosition = track.name.endsWith(".position");
            if (!isPosition) return true; 
            const isHips = track.name.includes("Hips") || track.name.includes("mixamorigHips");
            return !isHips; // Remove hips position -> In Place
        });
      } else {
        cloned.tracks = cloned.tracks.filter((track) => {
            const isPosition = track.name.endsWith(".position");
            const isHips = track.name.includes("Hips") || track.name.includes("mixamorigHips");
            if (isPosition && !isHips) return false;
            return true;
        });
      }
      cloned.tracks.forEach((track) => {
        track.name = track.name.replace("mixamorig", "");
        if (track.name.endsWith(".position")) {
           for (let i = 0; i < track.values.length; i++) {
             track.values[i] *= 0.01; 
           }
        }
      });
      return cloned;
    };
    return [
      cleanAnimation(idleData[0], "idle"),
      cleanAnimation(talkData[0], "talk"),
      cleanAnimation(transData[0], "transition"),
      cleanAnimation(confusedData[0], "confused"),
      cleanAnimation(walkLeftData[0], "walkLeft"),
      cleanAnimation(walkRightData[0], "walkRight")
    ];
  }, [idleData, talkData, transData, confusedData, walkLeftData, walkRightData]);

  const { actions } = useAnimations(animations, group);
  const [animation, setAnimation] = useState("idle")
  const [lipsync, setLipsync] = useState(null)

  const audioContextRef = useRef(null);
  const streamDestinationRef = useRef(null);
  const audioRef = useRef(new Audio()); 
  const lastAudioUrlRef = useRef(null);

  useEffect(() => {
    if (!audioContextRef.current) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContextRef.current = new AudioContext({ latencyHint: 'interactive' });
      streamDestinationRef.current = audioContextRef.current.createMediaStreamDestination();
      const source = audioContextRef.current.createMediaElementSource(audioRef.current);
      source.connect(streamDestinationRef.current);
      const silentGain = audioContextRef.current.createGain();
      silentGain.gain.value = 1.0; 
      source.connect(silentGain);
      silentGain.connect(audioContextRef.current.destination);
      if (onAudioTrackReady) onAudioTrackReady(streamDestinationRef.current.stream.getAudioTracks()[0]);
    }
    const resumeAudio = () => { if (audioContextRef.current?.state === 'suspended') audioContextRef.current.resume(); };
    document.addEventListener('click', resumeAudio);
    return () => document.removeEventListener('click', resumeAudio);
  }, []);

  // --- LOGIC 1: MOVEMENT (STRICT PRIORITY) ---
  useEffect(() => {
    if (sequenceState === "WALK_OFF") {
        setAnimation("walkLeft");
        return;
    } 
    if (sequenceState === "WALK_ON") {
        setAnimation("walkRight");
        return;
    }
    if (sequenceState === "IDLE") {
        if (animation === "walkLeft" || animation === "walkRight") {
            setAnimation("idle");
        }
    }
  }, [sequenceState]);

  // --- LOGIC 2: TALKING ---
  useEffect(() => {
    if (sequenceState !== "IDLE") return;

    if (status === "SPEAKING" && audioUrl && visemeUrl) {
      if (audioUrl === lastAudioUrlRef.current) return;
      lastAudioUrlRef.current = audioUrl; 

      if (audioContextRef.current) audioContextRef.current.resume();
      const audioEl = audioRef.current;
      audioEl.src = audioUrl;
      audioEl.crossOrigin = "anonymous";

      fetch(visemeUrl)
        .then(res => res.json())
        .then(json => {
          setLipsync(json);
          // Only change animation if we are currently Idle (don't interrupt walk)
          if (gesture === "confused") setAnimation("confused");
          else if (gesture === "transition") setAnimation("transition");
          else setAnimation("talk");
          
          audioEl.play().catch(e => console.error("Playback failed:", e));
        })
        .catch(err => console.error("Viseme Fetch Error:", err)); // Added catch

      audioEl.onended = () => {
        setAnimation("idle");
        setLipsync(null);
        if (onSpeechEnded) onSpeechEnded(); 
      };
    }
  }, [status, audioUrl, visemeUrl, gesture, sequenceState]); 

  // --- ANIMATION MIXER ---
  useEffect(() => {
    if (!actions || !animation || !group.current) return
    const action = actions[animation];
    action.reset().fadeIn(0.2).play();
    return () => { 
        action.fadeOut(0.2);
    }
  }, [actions, animation])

  useFrame((state, delta) => {
    const audio = audioRef.current;
    if (!lipsync || !audio || audio.paused || audio.ended) return;
    const time = audio.currentTime;
    let activeCue = null;
    for (let cue of lipsync.mouthCues) {
      if (time >= cue.start && time <= cue.end) { activeCue = cue; break; }
    }
    let currentViseme = "viseme_sil";
    let cueProgress = 0;
    if (activeCue) {
      const mapped = correspondings[activeCue.value];
      if (mapped) currentViseme = mapped;
      const span = Math.max(0.0001, activeCue.end - activeCue.start);
      cueProgress = (time - activeCue.start) / span;
    }
    const easeInOut = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const visemePulse = easeInOut(Math.min(Math.max(cueProgress, 0), 1));
    const smoothFactor = 1 - Math.exp(-delta * 20);
    const jawSmoothFactor = 1 - Math.exp(-delta * 10);
    const allVisemes = ["viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD", "viseme_kk", "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR", "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U", "viseme_sil"];
    [nodes.Head_Mesh, nodes.Teeth_Mesh, nodes.Tongue_Mesh].forEach((mesh) => {
      if (!mesh) return;
      allVisemes.forEach((visemeName) => {
        const isActive = visemeName === currentViseme;
        const target = isActive ? visemePulse : 0.0;
        const idx = mesh.morphTargetDictionary[visemeName];
        if (idx !== undefined) mesh.morphTargetInfluences[idx] = lerp(mesh.morphTargetInfluences[idx] || 0, target, smoothFactor);
      });
      const jawBase = jawOpenByViseme[currentViseme] ?? 0;
      const jawTarget = jawBase * visemePulse;
      const jawIdx = mesh.morphTargetDictionary["jawOpen"];
      if (jawIdx !== undefined) mesh.morphTargetInfluences[jawIdx] = lerp(mesh.morphTargetInfluences[jawIdx] || 0, jawTarget, jawSmoothFactor);
    });
  });
  const lerp = (a, b, t) => a + (b - a) * t;

  return (
    <group {...props} dispose={null} ref={group} visible={visible}>
      <primitive object={clone} />
      {nodes.Body_Mesh && <skinnedMesh frustumCulled={false} geometry={nodes.Body_Mesh.geometry} material={materials.Body} skeleton={nodes.Body_Mesh.skeleton} />}
      {nodes.avaturn_hair_0 && <skinnedMesh frustumCulled={false} geometry={nodes.avaturn_hair_0.geometry} material={materials.avaturn_hair_0_material} skeleton={nodes.avaturn_hair_0.skeleton} />}
      {nodes.avaturn_hair_1 && <skinnedMesh frustumCulled={false} geometry={nodes.avaturn_hair_1.geometry} material={materials.avaturn_hair_1_material} skeleton={nodes.avaturn_hair_1.skeleton} />}
      {nodes.avaturn_shoes_0 && <skinnedMesh frustumCulled={false} geometry={nodes.avaturn_shoes_0.geometry} material={materials.avaturn_shoes_0_material} skeleton={nodes.avaturn_shoes_0.skeleton} />}
      {nodes.avaturn_look_0 && <skinnedMesh frustumCulled={false} geometry={nodes.avaturn_look_0.geometry} material={materials.avaturn_look_0_material} skeleton={nodes.avaturn_look_0.skeleton} />}
      {nodes.Eye_Mesh && <skinnedMesh frustumCulled={false} name="Eye_Mesh" geometry={nodes.Eye_Mesh.geometry} material={materials.Eyes} skeleton={nodes.Eye_Mesh.skeleton} morphTargetDictionary={nodes.Eye_Mesh.morphTargetDictionary} morphTargetInfluences={nodes.Eye_Mesh.morphTargetInfluences} />}
      {nodes.EyeAO_Mesh && <skinnedMesh frustumCulled={false} name="EyeAO_Mesh" geometry={nodes.EyeAO_Mesh.geometry} material={materials.EyeAO} skeleton={nodes.EyeAO_Mesh.skeleton} morphTargetDictionary={nodes.EyeAO_Mesh.morphTargetDictionary} morphTargetInfluences={nodes.EyeAO_Mesh.morphTargetInfluences} />}
      {nodes.Eyelash_Mesh && <skinnedMesh frustumCulled={false} name="Eyelash_Mesh" geometry={nodes.Eyelash_Mesh.geometry} material={materials.Eyelash} skeleton={nodes.Eyelash_Mesh.skeleton} morphTargetDictionary={nodes.Eyelash_Mesh.morphTargetDictionary} morphTargetInfluences={nodes.Eyelash_Mesh.morphTargetInfluences} />}
      {nodes.Head_Mesh && <skinnedMesh frustumCulled={false} name="Head_Mesh" geometry={nodes.Head_Mesh.geometry} material={materials.Head} skeleton={nodes.Head_Mesh.skeleton} morphTargetDictionary={nodes.Head_Mesh.morphTargetDictionary} morphTargetInfluences={nodes.Head_Mesh.morphTargetInfluences} />}
      {nodes.Teeth_Mesh && <skinnedMesh frustumCulled={false} name="Teeth_Mesh" geometry={nodes.Teeth_Mesh.geometry} material={materials.Teeth} skeleton={nodes.Teeth_Mesh.skeleton} morphTargetDictionary={nodes.Teeth_Mesh.morphTargetDictionary} morphTargetInfluences={nodes.Teeth_Mesh.morphTargetInfluences} />}
      {nodes.Tongue_Mesh && <skinnedMesh frustumCulled={false} name="Tongue_Mesh" geometry={nodes.Tongue_Mesh.geometry} material={materials.Teeth} skeleton={nodes.Tongue_Mesh.skeleton} morphTargetDictionary={nodes.Tongue_Mesh.morphTargetDictionary} morphTargetInfluences={nodes.Tongue_Mesh.morphTargetInfluences} />}
      {nodes.Glasses_Mesh && <skinnedMesh frustumCulled={false} geometry={nodes.Glasses_Mesh.geometry} material={materials.Glasses} skeleton={nodes.Glasses_Mesh.skeleton} />}
    </group>
  )
}

// FIX: Preload everything to prevent "Blanks" (Suspense fallbacks) during runtime
useGLTF.preload('/models/wyy.glb')
useFBX.preload("animations/boy/Idle.fbx")
useFBX.preload("animations/boy/Talking_onefingerexplain(default).fbx")
useFBX.preload("animations/boy/Talking_onepalmup(gesture-in-out).fbx")
useFBX.preload("animations/boy/Talking_doublepalmup(na).fbx")
useFBX.preload("animations/boy/Left Strafe Walk.fbx")
useFBX.preload("animations/boy/Right Strafe Walk.fbx")