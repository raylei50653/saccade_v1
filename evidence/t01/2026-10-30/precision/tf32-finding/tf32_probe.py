import os, torch, torch.nn.functional as F
def probe():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, 64, 32, 32, device="cuda", generator=g); w = torch.randn(64, 64, 3, 3, device="cuda", generator=g)
    a = torch.randn(256, 512, device="cuda", generator=g); b = torch.randn(512, 256, device="cuda", generator=g)
    rc = F.conv2d(x.double(), w.double(), padding=1); c = F.conv2d(x, w, padding=1)
    rm = a.double() @ b.double(); m = a @ b
    return float((c.double()-rc).norm()/rc.norm()), float((m.double()-rm).norm()/rm.norm())
def state():
    return dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
      mm_prec=torch.get_float32_matmul_precision(), cudnn_all=torch.backends.cudnn.fp32_precision, cudnn_conv=torch.backends.cudnn.conv.fp32_precision,
      cuda_mm=torch.backends.cuda.matmul.fp32_precision, generic=torch.backends.fp32_precision)
print("env", os.environ.get("NVIDIA_TF32_OVERRIDE"))
print("default", state(), probe())
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
print("legacy-off", state(), probe())
torch.backends.cudnn.allow_tf32 = True
print("cudnn-on", state(), probe())
