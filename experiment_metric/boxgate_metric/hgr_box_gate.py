import numpy as np

class BoxGate:
    def __init__(self):
        # CONFIGURATION
        self.MIN_FRAMES = 8
        self.BOX_RADIUS = 0.08      # Stricter Swipe (8cm)
        self.SCALE_THRESH = 0.01   # Stricter Grab
        self.stop_frame = 0
        
        # STATE
        self.is_recording = False
        self.anchor_pos = None      
        self.anchor_scale = None    
        self.frame_count = 0
        self.stop_counter = 0
        
        # Ignore the first 7 frames of a new hand to prevent "Entry Bursts"
        self.warmup_counter = 7

    def process(self, lmCoords):
        """
        Input: 3D Landmarks
        """
        curr_pos = lmCoords[0] # Wrist
        
        # Calculate Scale
        tips = lmCoords[[4,8,12,16,20]]
        curr_scale = np.mean(np.linalg.norm(tips - curr_pos, axis=1))
        
        state = "IDLE"
        debug_val = 0.0
        
        #  LOGIC A: WARMUP STAGE  ## Hard-lock the anchor to the hand (prevent triggering)
        if self.warmup_counter > 0:
            self.warmup_counter -= 1
            self.anchor_pos = curr_pos ## current position of the wrist (in 3d scale)
            self.anchor_scale = curr_scale ## current scaling we have 
            return "WARMUP", 0.0

        #  LOGIC B: IDLE STAGE 
        if not self.is_recording:
            # Init Anchor if missing (should be set by warmup, but safety check)
            if self.anchor_pos is None:
                self.anchor_pos = curr_pos
                self.anchor_scale = curr_scale
            
            # Check Diffs 
            dist_moved = np.linalg.norm(curr_pos - self.anchor_pos) ## for swipe case
            scale_change = abs(curr_scale - self.anchor_scale) ## for grab case
            
            ## Trigger based on current distance or scale change based on thier previous anchor
            # 1. TRIGGER: Swipe
            if dist_moved > self.BOX_RADIUS:
                self.is_recording = True
                self.frame_count = 1
                state = "RECORDING"

            # 2. TRIGGER: Grab
            elif scale_change > self.SCALE_THRESH:
                self.is_recording = True
                self.frame_count = 1
                state = "RECORDING"
            
            # 3. SMART ANCHOR (Drift Correction)
            else:
                if dist_moved < (self.BOX_RADIUS * 0.5):
                    # Center safe zone -> Quick follow
                    self.anchor_pos = (0.90 * self.anchor_pos) + (0.10 * curr_pos)
                    self.anchor_scale = (0.90 * self.anchor_scale) + (0.10 * curr_scale)
                else:
                    # Edge zone -> Resist drift
                    self.anchor_pos = (0.98 * self.anchor_pos) + (0.02 * curr_pos)
                    self.anchor_scale = (0.98 * self.anchor_scale) + (0.02 * curr_scale)

            debug_val = dist_moved

        #  LOGIC C: RECORDING STAGE
        else:
            self.frame_count += 1
            state = "RECORDING"
            
            # Stability Check
            step_move = 0.0
            if hasattr(self, 'prev_pos'):
                step_move = np.linalg.norm(curr_pos - self.prev_pos)
            step_scale = 0.0
            if hasattr(self, 'prev_scale'):
                step_scale = abs(curr_scale - self.prev_scale)
            
            # Strict Stop (Must be very still)
            # if step_move < 0.01: 
            #     self.stop_counter += 1
            # else:
            #     self.stop_counter = 0 
            is_wrist_still = step_move < 0.01
            is_hand_still  = step_scale < 0.001

            if is_wrist_still and is_hand_still:
                self.stop_counter += 1
            else:
                self.stop_counter = 0 # Reset if EITHER is moving
            
            # we need total consecutive 8 frame to confirm that our stop condition has reached
            if self.stop_counter > self.stop_frame:
                self.is_recording = False

                # Reset anchor HERE
                self.anchor_pos = curr_pos 
                self.anchor_scale = curr_scale
                
                if self.frame_count >= self.MIN_FRAMES:
                    state = "FINISHED"
                else:
                    state = "REJECTED"
            
            debug_val = step_move

        self.prev_pos = curr_pos
        self.prev_scale = curr_scale
        return state, debug_val

    ## This is where we need to reset on loss of hand
    def reset(self):
        self.is_recording = False
        self.anchor_pos = None
        self.anchor_scale = None
        self.stop_counter = 0
        self.warmup_counter = 7 # Reset back 