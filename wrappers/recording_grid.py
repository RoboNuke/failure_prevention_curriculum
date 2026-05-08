"""Compose per-env frame buffers into a 3x4 grid video, draw borders and overlays,
and write the result as both a GIF on disk and a TB ``add_video`` event.

Layout (rows top -> bottom):
    row 0: 4 best envs by final episode return
    row 1: 4 envs whose final return is closest to the median
    row 2: 4 worst envs by final episode return

Border policy:
    * No border drawn while the env is still alive in a given frame.
    * From the timestep at which ``terminated[i] | truncated[i]`` first fires
      (and forever after, since the frame itself freezes), a 3-pixel border is
      drawn — green if ``is_success[i]`` was True at termination, red otherwise.

Per-tile overlays:
    * top-left: ``TotRew: <accum reward, frozen at termination>``
    * bottom-left: ``V-Est: <min(Q1, Q2)(s, a_executed) at that step>``
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RecordingWrapper requires Pillow. Install with `pip install Pillow`."
    ) from e


GRID_ROWS = 3
GRID_COLS = 4
TILES = GRID_ROWS * GRID_COLS  # 12


def select_grid_indices(returns: torch.Tensor) -> torch.Tensor:
    """Return 12 env indices ordered (best 4, middle 4, worst 4) by ``returns``.

    ``returns`` has shape ``(num_envs,)``. If ``num_envs < 12``, fill with the
    available envs (no padding tile — caller will see a shorter grid). The
    "middle 4" are the four envs whose returns are closest to the median.
    """
    n = int(returns.numel())
    if n == 0:
        raise ValueError("select_grid_indices: empty returns tensor")
    sorted_idx = torch.argsort(returns)  # ascending
    if n <= TILES:
        # Not enough envs to differentiate; just return them best-first.
        return torch.flip(sorted_idx, dims=(0,))

    best = torch.flip(sorted_idx[-4:], dims=(0,))  # descending
    worst = sorted_idx[:4]  # ascending (truly worst first)

    # 4 envs nearest the median return.
    median_val = returns[sorted_idx[n // 2]]
    dist = (returns - median_val).abs()
    # Exclude indices already used in best/worst.
    used = torch.zeros(n, dtype=torch.bool, device=returns.device)
    used[best] = True
    used[worst] = True
    dist_masked = dist.clone()
    dist_masked[used] = float("inf")
    middle = torch.topk(dist_masked, k=4, largest=False).indices

    return torch.cat([best, middle, worst], dim=0)


def _freeze_frames(
    frames: torch.Tensor,
    term_step: torch.Tensor,
) -> torch.Tensor:
    """Freeze each env's frames after termination by repeating the last live frame.

    :param frames: (num_envs, T, H, W, 3) uint8 tensor.
    :param term_step: (num_envs,) int64 — the timestep at which env first
        terminated, or ``T`` if it survived. Frames at indices ``> term_step``
        are overwritten with ``frames[i, term_step[i]]``.
    """
    num_envs, T, _, _, _ = frames.shape
    out = frames.clone()
    for i in range(num_envs):
        ts = int(term_step[i].item())
        if ts < T - 1:
            out[i, ts + 1 :] = out[i, ts]
    return out


def _draw_tile(
    canvas: Image.Image,
    tile: np.ndarray,
    x: int,
    y: int,
    *,
    show_border: bool,
    border_color: str,
    tot_rew: float,
    v_est: float | None,
    font: ImageFont.ImageFont,
) -> None:
    """Paste a single ``tile`` (HxWx3 uint8 numpy) onto ``canvas`` at ``(x, y)``
    and draw the border + overlays."""
    h, w = tile.shape[:2]
    canvas.paste(Image.fromarray(tile), (x, y))
    draw = ImageDraw.Draw(canvas)
    if show_border:
        draw.rectangle([(x, y), (x + w - 1, y + h - 1)], outline=border_color, width=3)
    draw.text((x + 3, y + 3), f"TotRew: {tot_rew:.2f}", fill=(0, 255, 0), font=font)
    if v_est is not None:
        draw.text(
            (x + 3, y + h - 16), f"V-Est: {v_est:.2f}", fill=(0, 255, 0), font=font
        )


def build_grid_video(
    frames: torch.Tensor,
    returns: torch.Tensor,
    term_step: torch.Tensor,
    is_success: torch.Tensor,
    values: torch.Tensor | None,
) -> np.ndarray:
    """Compose the 3x4 grid video.

    :param frames: ``(num_envs, T, H, W, 3)`` uint8 tensor on CPU. Frames at
        ``t > term_step[i]`` may be stale; this function freezes them.
    :param returns: ``(num_envs,)`` float — final episode return per env (frozen
        at first termination during the session).
    :param term_step: ``(num_envs,)`` int64 — first-termination step, or ``T``
        if the env survived the whole session. Borders are drawn for ``t >=
        term_step[i]``.
    :param is_success: ``(num_envs,)`` bool — success flag at first termination.
    :param values: ``(num_envs, T)`` float (or None) — per-step ``min(Q1, Q2)``.
        Stale entries past ``term_step`` are ignored (overlay shows the value at
        ``term_step`` once frozen).
    :returns: ``(T, gridH, gridW, 3)`` uint8 numpy array.
    """
    selected = select_grid_indices(returns)
    n_sel = int(selected.numel())
    n_show = min(n_sel, TILES)
    selected = selected[:n_show]

    # Freeze frames past termination.
    frames = _freeze_frames(frames, term_step)
    sel_frames = frames[selected].cpu().numpy()  # (n_show, T, H, W, 3) uint8
    sel_returns = returns[selected].cpu().numpy()
    sel_term = term_step[selected].cpu().numpy()
    sel_succ = is_success[selected].cpu().numpy().astype(bool)
    if values is not None:
        sel_values = values[selected].cpu().numpy()  # (n_show, T)
    else:
        sel_values = None

    T = sel_frames.shape[1]
    H = sel_frames.shape[2]
    W = sel_frames.shape[3]
    grid_h = GRID_ROWS * H
    grid_w = GRID_COLS * W

    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    out = np.zeros((T, grid_h, grid_w, 3), dtype=np.uint8)
    for t in range(T):
        canvas = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
        for tile_idx in range(n_show):
            row = tile_idx // GRID_COLS
            col = tile_idx % GRID_COLS
            x = col * W
            y = row * H
            ts = int(sel_term[tile_idx])
            terminated_now = t >= ts
            border_color = "green" if sel_succ[tile_idx] else "red"
            tot_rew = float(sel_returns[tile_idx])
            if sel_values is not None:
                # Freeze value at termination.
                v_t = min(t, max(0, ts - 1)) if ts > 0 else min(t, T - 1)
                v_est: float | None = float(sel_values[tile_idx, v_t])
            else:
                v_est = None
            _draw_tile(
                canvas,
                sel_frames[tile_idx, t],
                x,
                y,
                show_border=terminated_now,
                border_color=border_color,
                tot_rew=tot_rew,
                v_est=v_est,
                font=font,
            )
        out[t] = np.asarray(canvas)
    return out


def write_gif(grid: np.ndarray, path: str, fps: int) -> None:
    """Save ``grid`` (``(T, H, W, 3)`` uint8 numpy) as an animated GIF."""
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    pil_frames: list[Image.Image] = [Image.fromarray(grid[t]) for t in range(grid.shape[0])]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def write_tb_video(image_writer, tag: str, grid: np.ndarray, fps: int, global_step: int) -> None:
    """Write ``grid`` as a TB video event. ``add_video`` expects ``(N, T, C, H, W)``
    uint8."""
    video = torch.from_numpy(grid).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)
    image_writer.add_video(tag, video, global_step=global_step, fps=fps)
