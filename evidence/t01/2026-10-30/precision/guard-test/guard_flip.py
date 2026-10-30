import time, torch
torch.backends.cudnn.allow_tf32 = True   # a library flipping the flag mid-run
time.sleep(25)
print("SHOULD NOT REACH")
