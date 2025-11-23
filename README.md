# My GTX-750 Brain: Alive, Talking, and Growing

I’m the guy who refuses to retire a ten-year-old GTX-750 (Maxwell, CC 5.0, ~500 GFLOPS). 

With CUDA 12 + cupy-cuda11x I’ve turned it into a real-time spiking brain that you can literally chat with - well sort of. 

# What’s Already Running   
- 200 000 leaky integrate-and-fire neurons, 100 % on GPU  
- Sparse connectivity: 32 random outgoing synapses per neuron (CuPy sparse matrix + cp.roll for local neighborhoods)  
- Full homeostasis – global drive + adaptive thresholds keep firing pinned at ~5 % forever  
- Real-time stepping: 0.017 s per simulation step on this ancient card  
- Continuous mode – no more “run-once-and-die”; waves and ripples roll indefinitely  
- Lightweight logging (spike count, mean input, mean threshold) → brain_debug.log without killing performance  
- Interactive text loop:  
  You type → text → stimulus vector → external current injected → ~200 steps → raw spikes → ASCII decode → brain replies  
  Examples I’ve tested dozens of times:  
  “hello” → “ooooooo”  
  “cat” → “c??o?a?t”  
  “why?” → “?oo????o”  
  “123” → “3111?.?2”  
  Gibberish, yes – but repeatable gibberish = the I/O mapping works perfectly  
- Persistent memory: synaptic weights saved/loaded from disk + slow Hebbian drift between sessions  

In short: I have a primitive, self-regulating, GPU-accelerated organism that never flatlines, learns a tiny bit, and answers when I talk to it.

# Next Upgrades   
A. Attractor ensembles – turn random spike clouds into stable letter/word clusters (first step to coherent replies)  
B. Proper reward-based plasticity – I tell it “good” or “bad” and synapses actually strengthen/weaken accordingly  
C. 2D cortical sheet (500×500 layout) with dedicated input/output patches – real topography instead of soup  
D. Short-term memory loops – eligibility traces + recurrent gating so it can finish a “thought”

Just say the letter. I’m at the keyboard, GPU is warm, and the brain is waiting for its next evolutionary jump.

