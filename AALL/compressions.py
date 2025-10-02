import jax
import jax.numpy as jnp
from matplotlib.widgets import Slider

def _rect_mask(shape_hw, keep_rows: int, keep_cols: int, *, centered: bool) -> jnp.ndarray:
    """Build a boolean mask (H,W) with a kept rectangle either top-left or centered."""
    H, W = shape_hw
    kr = int(jnp.clip(keep_rows, 0, H))
    kc = int(jnp.clip(keep_cols, 0, W))

    mask = jnp.zeros((H, W), dtype=bool)
    if centered:
        r0 = H // 2 - kr // 2
        c0 = W // 2 - kc // 2
        r1, c1 = r0 + kr, c0 + kc
    else:
        r0, c0 = 0, 0
        r1, c1 = kr, kc

    # Guard against negatives if kr/kc are 0
    r0 = jnp.clip(r0, 0, H); r1 = jnp.clip(r1, 0, H)
    c0 = jnp.clip(c0, 0, W); c1 = jnp.clip(c1, 0, W)

    # Fill the rectangle
    mask = mask.at[r0:r1, c0:c1].set(True)
    return mask

def _apply_mask(x: jnp.ndarray, mask_hw: jnp.ndarray) -> jnp.ndarray:
    """Broadcast (H,W) mask to (B,H,W) or (B,H,W,C) and apply."""
    if x.ndim == 3:
        return x * mask_hw[None, :, :]
    elif x.ndim == 4:
        return x * mask_hw[None, :, :, None]
    else:
        raise ValueError(f"Expected 3D or 4D, got {x.ndim}D")
    
def dft2_batch(images: jnp.ndarray, *, center: bool = True, norm: str = "backward", keep_rows: int | None = None, keep_cols: int | None = None) -> jnp.ndarray:
    """
    2-D DFT over spatial axes (H, W) for a batch of images.
    - images: (B,H,W) or (B,H,W,C) array; real or complex.
    - center: if True, return frequency spectrum shifted to center (fftshift).
    - norm: 'backward' | 'ortho' | 'forward' (mirrors scipy.fft).
    Returns complex array of same shape as input.
    """
    if images.ndim not in (3, 4):
        raise ValueError(f"Expected (B,H,W) or (B,H,W,C), got shape {images.shape}")

    F = jax.numpy.fft.fft2(images, axes=(1, 2), norm=norm)
    if center:
        F = jax.numpy.fft.fftshift(F, axes=(1, 2))

    if (keep_rows is not None) and (keep_cols is not None):
        mask = _rect_mask(F.shape[1:3], keep_rows, keep_cols, centered=center)
        F = _apply_mask(F, mask)
    
    return F

def idft2_batch(freq: jnp.ndarray, *, centered: bool = True, norm: str = "backward") -> jnp.ndarray:
    """
    Inverse 2-D DFT over (H, W) for a batch of spectra.
    - freq: (B,H,W) or (B,H,W,C) complex array from dft2_batch.
    - centered: True if `freq` is already fftshift-centered (apply ifftshift first).
    - norm: must match what was used in dft2_batch.
    Returns complex array; use jnp.real(...) if original inputs were real.
    """
    if freq.ndim not in (3, 4):
        raise ValueError(f"Expected (B,H,W) or (B,H,W,C), got shape {freq.shape}")

    F = jax.numpy.fft.ifftshift(freq, axes=(1, 2)) if centered else freq
    x = jax.numpy.fft.ifft2(F, axes=(1, 2), norm=norm)
    return x

def dct2_batch(images: jnp.ndarray, *, norm: str = "ortho", keep_rows: int | None = None, keep_cols: int | None = None) -> jnp.ndarray:
    """
    Apply 2-D DCT-II over spatial axes (H, W) for a batch of images.
    - images: (B,H,W) or (B,H,W,C), real array.
    - norm: None | 'ortho' (usually 'ortho' for unitary transform).
    Returns real array of same shape.
    """
    if images.ndim not in (3, 4):
        raise ValueError(f"Expected (B,H,W) or (B,H,W,C), got shape {images.shape}")

    # DCT along height (axis=1), then width (axis=2)
    x = jax.scipy.fft.dctn(images, type=2, axes=(1, 2), norm=norm)

    if (keep_rows is not None) and (keep_cols is not None):
        mask = _rect_mask(x.shape[1:3], keep_rows, keep_cols, centered=False)
        x = _apply_mask(x, mask)

    return x

def idct2_batch(coeffs: jnp.ndarray, *, norm: str = "ortho") -> jnp.ndarray:
    """
    Inverse 2-D DCT (i.e. DCT-III) over spatial axes (H, W).
    - coeffs: (B,H,W) or (B,H,W,C) real array from dct2_batch.
    - norm: must match what was used in dct2_batch.
    Returns real array of same shape.
    """
    if coeffs.ndim not in (3, 4):
        raise ValueError(f"Expected (B,H,W) or (B,H,W,C), got shape {coeffs.shape}")

    # IDCT along width then height
    x = jax.scipy.fft.idctn(coeffs, type=2, axes=(1, 2), norm=norm)
    return x

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- your functions (dft2_batch, idft2_batch, dct2_batch, idct2_batch) go here ---
def to_float01(arr: jnp.ndarray) -> jnp.ndarray:
    """Normalize image to float32 in [0,1] for display only."""
    x = jnp.asarray(arr)

    # If uint8, just scale
    if x.dtype == jnp.uint8:
        return x.astype(jnp.float32) / 255.0

    x = x.astype(jnp.float32)

    # Heuristics:
    # - If values look like 0..255, scale by 255
    # - Else if already in 0..1 (or slightly out due to numerics), clip
    mx = jnp.nanmax(x)
    mn = jnp.nanmin(x)

    # Scale 0..255 floats down to 0..1 if max is clearly > 1
    if mx > 1.5:
        x = x / 255.0

    # Clip small negatives / >1 due to rounding/IDFT tiny errors
    x = jnp.clip(x, 0.0, 1.0)
    return x

def show_images(original, reconstructed, title=""):
    plt.figure(figsize=(8,4))
    plt.suptitle(title)

    # Original
    plt.subplot(1,2,1)
    plt.imshow(to_float01(original))
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    plt.subplot(1,2,2)
    plt.imshow(to_float01(reconstructed))
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

def test_rgb_image(img: jnp.ndarray):
    """
    img: single RGB image (H,W,3) or batch (B,H,W,3)
    Runs DFT2 and DCT2 with coefficient drop, shows results in one window.
    """
    if img.ndim == 3:
        img = img[None, ...]  # add batch axis
    img = jnp.array(img, dtype=float)

    # --- DFT with coefficient truncation ---
    F = dft2_batch(img, center=True, norm="ortho", keep_cols=30, keep_rows=30)
    rec_dft = jnp.real(idft2_batch(F, centered=True, norm="ortho"))

    # --- DCT with coefficient truncation ---
    C = dct2_batch(img, norm="ortho", keep_cols=30, keep_rows=30)
    rec_dct = idct2_batch(C, norm="ortho")

    # --- Plot all three ---
    plt.figure(figsize=(12,4))
    plt.suptitle("Original vs FFT vs DCT (coeff truncation)")

    plt.subplot(1,3,1)
    plt.imshow(to_float01(img[0]))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(to_float01(rec_dft[0]))
    plt.title("FFT Reconstructed")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(to_float01(rec_dct[0]))
    plt.title("DCT Reconstructed")
    plt.axis("off")

    plt.show()

def show_with_sliders(img: jnp.ndarray):
    if img.ndim == 3:
        img = img[None, ...]  # add batch axis
    img = jnp.array(img, dtype=float)
    H, W = img.shape[1:3]

    # Initial values
    init_keep_r, init_keep_c = H//4, W//4

    # Compute initial reconstructions
    def reconstruct(keep_r, keep_c):
        F = dft2_batch(img, center=True, norm="ortho",
                       keep_rows=keep_r, keep_cols=keep_c)
        rec_dft = jnp.real(idft2_batch(F, centered=True, norm="ortho"))

        C = dct2_batch(img, norm="ortho",
                       keep_rows=keep_r, keep_cols=keep_c)
        rec_dct = idct2_batch(C, norm="ortho")

        return rec_dft, rec_dct

    rec_dft, rec_dct = reconstruct(init_keep_r, init_keep_c)

    # --- Setup figure ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(bottom=0.25)  # make room for sliders

    im0 = axes[0].imshow(to_float01(img[0]))
    axes[0].set_title("Original"); axes[0].axis("off")

    im1 = axes[1].imshow(to_float01(rec_dft[0]))
    axes[1].set_title("FFT Reconstructed"); axes[1].axis("off")

    im2 = axes[2].imshow(to_float01(rec_dct[0]))
    axes[2].set_title("DCT Reconstructed"); axes[2].axis("off")

    # --- Add sliders ---
    ax_keep_r = plt.axes([0.25, 0.1, 0.55, 0.03])
    ax_keep_c = plt.axes([0.25, 0.05, 0.55, 0.03])

    s_keep_r = Slider(ax_keep_r, 'Rows', 1, H, valinit=init_keep_r, valstep=1)
    s_keep_c = Slider(ax_keep_c, 'Cols', 1, W, valinit=init_keep_c, valstep=1)

    # --- Update function ---
    def update(val):
        keep_r, keep_c = int(s_keep_r.val), int(s_keep_c.val)
        rec_dft, rec_dct = reconstruct(keep_r, keep_c)
        im1.set_data(to_float01(rec_dft[0]))
        im2.set_data(to_float01(rec_dct[0]))
        fig.canvas.draw_idle()

    s_keep_r.on_changed(update)
    s_keep_c.on_changed(update)

    plt.show()

    print("✅ FFT & DCT reconstructions displayed together.")
def test_dft2_roundtrip():
    imgs_gray = jnp.arange(2*8*8).reshape(2, 8, 8).astype(float)
    imgs_rgb  = jnp.arange(2*8*8*3).reshape(2, 8, 8, 3).astype(float)

    for imgs, name in [(imgs_gray, "DFT2 Gray"), (imgs_rgb, "DFT2 RGB")]:
        F = dft2_batch(imgs, center=True, norm="ortho")
        rec = idft2_batch(F, centered=True, norm="ortho")
        rec_real = jnp.real(rec)
        assert jnp.allclose(imgs, rec_real, atol=1e-6), "DFT roundtrip failed"
        show_images(imgs, rec_real, title=name)
    print("✅ DFT roundtrip passed and images displayed.")

def test_dct2_roundtrip():
    imgs_gray = jnp.arange(2*8*8).reshape(2, 8, 8).astype(float)
    imgs_rgb  = jnp.arange(2*8*8*3).reshape(2, 8, 8, 3).astype(float)

    for imgs, name in [(imgs_gray, "DCT2 Gray"), (imgs_rgb, "DCT2 RGB")]:
        C = dct2_batch(imgs, norm="ortho")
        rec = idct2_batch(C, norm="ortho")
        assert jnp.allclose(imgs, rec, atol=1e-6), "DCT roundtrip failed"
        show_images(imgs, rec, title=name)
    print("✅ DCT roundtrip passed and images displayed.")

if __name__ == "__main__":
    import matplotlib.image as mpimg
    # Load your image (replace with your file path)
    img = mpimg.imread("./AALL/anime-1_2k.png")   # shape (H,W,3) or (H,W,4) if RGBA
    if img.shape[-1] == 4:   # drop alpha channel if present
        img = img[..., :3]

    show_with_sliders(img)