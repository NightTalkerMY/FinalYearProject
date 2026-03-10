# # This version has issue with swiping
# import numpy as np

# class BoxGate:
#     def __init__(self):
#         # CONFIGURATION
#         self.MIN_FRAMES = 8
#         self.BOX_RADIUS = 0.08      # Stricter Swipe (8cm)
#         self.SCALE_THRESH = 0.055   # Stricter Grab
        
#         # STATE
#         self.is_recording = False
#         self.anchor_pos = None      
#         self.anchor_scale = None    
#         self.frame_count = 0
#         self.stop_counter = 0
        
#         # [NEW] SAFETY LOCK
#         # Ignore the first 20 frames of a new hand to prevent "Entry Bursts"
#         self.warmup_counter = 20

#     def process(self, lmCoords):
#         """
#         Input: 3D Landmarks
#         """
#         curr_pos = lmCoords[0] # Wrist
        
#         # Calculate Scale
#         tips = lmCoords[[4,8,12,16,20]]
#         curr_scale = np.mean(np.linalg.norm(tips - curr_pos, axis=1))
        
#         state = "IDLE"
#         debug_val = 0.0
        
#         # --- LOGIC A: WARMUP PHASE ---
#         if self.warmup_counter > 0:
#             self.warmup_counter -= 1
#             # Hard-lock the anchor to the hand (prevent triggering)
#             self.anchor_pos = curr_pos
#             self.anchor_scale = curr_scale
#             return "WARMUP", 0.0

#         # --- LOGIC B: IDLE (Inside the Box) ---
#         if not self.is_recording:
#             # Init Anchor if missing (should be set by warmup, but safety check)
#             if self.anchor_pos is None:
#                 self.anchor_pos = curr_pos
#                 self.anchor_scale = curr_scale
            
#             # Check Diffs
#             dist_moved = np.linalg.norm(curr_pos - self.anchor_pos)
#             scale_change = abs(curr_scale - self.anchor_scale)
            
#             # 1. TRIGGER: Swipe
#             if dist_moved > self.BOX_RADIUS:
#                 self.is_recording = True
#                 self.frame_count = 1
#                 state = "RECORDING"

#             # 2. TRIGGER: Grab
#             elif scale_change > self.SCALE_THRESH:
#                 self.is_recording = True
#                 self.frame_count = 1
#                 state = "RECORDING"
            
#             # 3. SMART ANCHOR (Drift Correction)
#             else:
#                 if dist_moved < (self.BOX_RADIUS * 0.5):
#                     # Center safe zone -> Quick follow
#                     self.anchor_pos = (0.90 * self.anchor_pos) + (0.10 * curr_pos)
#                     self.anchor_scale = (0.90 * self.anchor_scale) + (0.10 * curr_scale)
#                 else:
#                     # Edge zone -> Resist drift
#                     self.anchor_pos = (0.98 * self.anchor_pos) + (0.02 * curr_pos)
#                     self.anchor_scale = (0.98 * self.anchor_scale) + (0.02 * curr_scale)

#             debug_val = dist_moved

#         # --- LOGIC C: RECORDING ---
#         else:
#             self.frame_count += 1
#             state = "RECORDING"
            
#             # Stability Check
#             step_move = 0.0
#             if hasattr(self, 'prev_pos'):
#                 step_move = np.linalg.norm(curr_pos - self.prev_pos)
            
#             # Strict Stop (Must be very still)
#             if step_move < 0.01: 
#                 self.stop_counter += 1
#             else:
#                 self.stop_counter = 0 
            
#             if self.stop_counter > 8:
#                 self.is_recording = False
#                 self.anchor_pos = curr_pos # Reset anchor HERE
#                 self.anchor_scale = curr_scale
                
#                 if self.frame_count >= self.MIN_FRAMES:
#                     state = "FINISHED"
#                 else:
#                     state = "REJECTED"
            
#             debug_val = step_move

#         self.prev_pos = curr_pos
#         return state, debug_val

#     def reset(self):
#         self.is_recording = False
#         self.anchor_pos = None
#         self.stop_counter = 0
#         self.warmup_counter = 7 # Reset lock on hand loss


 # This latest version fix the early trigger stop for swiping 
import numpy as np

class BoxGate:
    def __init__(self):
        # CONFIGURATION
        self.MIN_FRAMES = 8
        self.BOX_RADIUS = 0.08      # 8cm
        self.SCALE_THRESH = 0.055   
        
        # STATE
        self.is_recording = False
        self.trigger_type = None    # [NEW] Tracks if we are Swiping or Grabbing
        self.anchor_pos = None      
        self.anchor_scale = None    
        self.frame_count = 0
        self.stop_counter = 0
        
        self.warmup_counter = 20

    def process(self, lmCoords):
        curr_pos = lmCoords[0] # Wrist
        
        # Calculate Scale
        tips = lmCoords[[4,8,12,16,20]]
        curr_scale = np.mean(np.linalg.norm(tips - curr_pos, axis=1))
        
        state = "IDLE"
        debug_val = 0.0
        
        # --- LOGIC A: WARMUP PHASE ---
        if self.warmup_counter > 0:
            self.warmup_counter -= 1
            self.anchor_pos = curr_pos
            self.anchor_scale = curr_scale
            return "WARMUP", 0.0

        # --- LOGIC B: IDLE (Inside the Box) ---
        if not self.is_recording:
            if self.anchor_pos is None:
                self.anchor_pos = curr_pos
                self.anchor_scale = curr_scale
            
            dist_moved = np.linalg.norm(curr_pos - self.anchor_pos)
            scale_change = abs(curr_scale - self.anchor_scale)
            
            # 1. TRIGGER: Swipe
            if dist_moved > self.BOX_RADIUS:
                self.is_recording = True
                self.trigger_type = "SWIPE"  # [NEW] Remember this!
                self.frame_count = 1
                state = "RECORDING"

            # 2. TRIGGER: Grab
            elif scale_change > self.SCALE_THRESH:
                self.is_recording = True
                self.trigger_type = "GRAB"   # [NEW] Remember this!
                self.frame_count = 1
                state = "RECORDING"
            
            # 3. SMART ANCHOR (Drift Correction)
            else:
                if dist_moved < (self.BOX_RADIUS * 0.5):
                    self.anchor_pos = (0.90 * self.anchor_pos) + (0.10 * curr_pos)
                    self.anchor_scale = (0.90 * self.anchor_scale) + (0.10 * curr_scale)
                else:
                    self.anchor_pos = (0.98 * self.anchor_pos) + (0.02 * curr_pos)
                    self.anchor_scale = (0.98 * self.anchor_scale) + (0.02 * curr_scale)

            debug_val = dist_moved

        # --- LOGIC C: RECORDING ---
        else:
            self.frame_count += 1
            state = "RECORDING"
            
            step_move = 0.0
            if hasattr(self, 'prev_pos'):
                step_move = np.linalg.norm(curr_pos - self.prev_pos)
            
            # [NEW] ADAPTIVE STOP LOGIC
            # ---------------------------------------------------------
            stop_threshold = 0.01   # Default Strict (1cm)
            max_stop_count = 8      # Default Quick Stop
            
            if self.trigger_type == "SWIPE":
                stop_threshold = 0.003  # [LOOSER] 3mm/frame (allows slow swipes)
                max_stop_count = 15     # [LONGER] Wait 0.5s before cutting off
            # ---------------------------------------------------------

            if step_move < stop_threshold: 
                self.stop_counter += 1
            else:
                self.stop_counter = 0 
            
            if self.stop_counter > max_stop_count:
                self.is_recording = False
                self.trigger_type = None # Reset
                
                self.anchor_pos = curr_pos 
                self.anchor_scale = curr_scale
                
                if self.frame_count >= self.MIN_FRAMES:
                    state = "FINISHED"
                else:
                    state = "REJECTED"
            
            debug_val = step_move

        self.prev_pos = curr_pos
        return state, debug_val

    def reset(self):
        self.is_recording = False
        self.trigger_type = None
        self.anchor_pos = None
        self.stop_counter = 0
        self.warmup_counter = 7