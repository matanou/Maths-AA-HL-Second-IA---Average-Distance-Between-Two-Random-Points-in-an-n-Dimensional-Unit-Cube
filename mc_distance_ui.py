#!/usr/bin/env python3
"""
Monte Carlo simulator for E[distance] between two random points in [0,1]^n,
with a friendly Tkinter UI and embedded Matplotlib histogram.

Features:
- Any dimension n (spinbox)
- Large N sampling with vectorized NumPy in batches
- Welford's online stats for mean/variance (numerically stable)
- Live progress bar, ETA, cancel
- Analytic checks for n=1 and n=2
- Export CSV (samples summary) and PNG (histogram)
"""

import threading
import queue
import time
import math
from datetime import timedelta
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ---------- Analytics: exact / known references ----------
def analytic_mean_n1():
    # E[|U-V|] where U,V ~ U(0,1) iid = 1/3
    return 1.0 / 3.0

def analytic_mean_n2():
    # E[D2] in unit square = (sqrt(2)+2+5 ln(1+sqrt2)) / 15
    return (math.sqrt(2) + 2 + 5 * math.log(1 + math.sqrt(2))) / 15


# ---------- Online stats (Welford) ----------
class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: np.ndarray):
        # batch update
        for v in x:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else float('nan')

    @property
    def std(self):
        return math.sqrt(self.variance) if self.n > 1 else float('nan')

    @property
    def sem(self):
        return self.std / math.sqrt(self.n) if self.n > 1 else float('nan')

    def summary(self):
        return {"n": self.n, "mean": self.mean, "std": self.std, "sem": self.sem}


# ---------- Worker thread for simulation ----------
class SimWorker(threading.Thread):
    def __init__(self, dim, total_samples, batch_size, seed, out_queue, cancel_event):
        super().__init__(daemon=True)
        self.dim = dim
        self.total = total_samples
        self.batch = batch_size
        self.seed = seed
        self.out = out_queue
        self.cancel = cancel_event

    def run(self):
        rng = np.random.default_rng(self.seed) if self.seed != "" else np.random.default_rng()
        stats = OnlineStats()
        hist_collect = []  # small subsample for plotting (to avoid memory explosion)

        start = time.time()
        processed = 0
        try:
            while processed < self.total and not self.cancel.is_set():
                b = min(self.batch, self.total - processed)
                # Vectorized sampling in [0,1]^dim
                U = rng.random((b, self.dim), dtype=np.float64)
                V = rng.random((b, self.dim), dtype=np.float64)
                diff = U - V
                # Euclidean norms
                d = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64))
                stats.update(d)

                # Subsample for histogram (cap ~100k)
                if len(hist_collect) < 100000:
                    take = min(b, 100000 - len(hist_collect))
                    hist_collect.extend(d[:take].tolist())

                processed += b
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else float('inf')
                remaining = self.total - processed
                eta = remaining / rate if rate > 0 else float('inf')

                # Send progress update
                self.out.put({
                    "type": "progress",
                    "processed": processed,
                    "total": self.total,
                    "elapsed": elapsed,
                    "eta": eta,
                    "mean": stats.mean,
                    "sem": stats.sem
                })

            # Done or canceled
            elapsed = time.time() - start
            self.out.put({
                "type": "done",
                "canceled": self.cancel.is_set(),
                "stats": stats.summary(),
                "elapsed": elapsed,
                "hist": np.array(hist_collect, dtype=np.float64)
            })

        except Exception as e:
            self.out.put({"type": "error", "message": str(e)})


# ---------- Main UI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Monte Carlo: Expected Distance in [0,1]^n")
        self.geometry("980x720")
        self.minsize(900, 640)
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except:
            pass

        # State
        self.worker = None
        self.q = queue.Queue()
        self.cancel_event = threading.Event()

        # Build UI
        self._build_controls()
        self._build_results()
        self._build_plot()

        self._poll_queue()

    def _build_controls(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(side=tk.TOP, fill=tk.X)

        # Dimension
        ttk.Label(frm, text="Dimension n:").grid(row=0, column=0, sticky="w")
        self.var_dim = tk.IntVar(value=2)
        spn_dim = ttk.Spinbox(frm, from_=1, to=1000, width=6, textvariable=self.var_dim)
        spn_dim.grid(row=0, column=1, padx=(4, 16))

        # Total samples
        ttk.Label(frm, text="Samples N:").grid(row=0, column=2, sticky="w")
        self.var_samples = tk.IntVar(value=200000)
        ent_samples = ttk.Entry(frm, width=12, textvariable=self.var_samples)
        ent_samples.grid(row=0, column=3, padx=(4, 16))

        # Batch size
        ttk.Label(frm, text="Batch size:").grid(row=0, column=4, sticky="w")
        self.var_batch = tk.IntVar(value=20000)
        ent_batch = ttk.Entry(frm, width=10, textvariable=self.var_batch)
        ent_batch.grid(row=0, column=5, padx=(4, 16))

        # Seed
        ttk.Label(frm, text="Random seed (optional):").grid(row=0, column=6, sticky="w")
        self.var_seed = tk.StringVar(value="")
        ent_seed = ttk.Entry(frm, width=12, textvariable=self.var_seed)
        ent_seed.grid(row=0, column=7, padx=(4, 16))

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=0, column=8, padx=(8, 0))

        self.btn_run = ttk.Button(btn_frame, text="Run Simulation", command=self._run)
        self.btn_run.pack(side=tk.LEFT, padx=4)

        self.btn_cancel = ttk.Button(btn_frame, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=4)

        self.btn_export = ttk.Button(btn_frame, text="Export CSV", command=self._export_csv, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=4)

        self.btn_savefig = ttk.Button(btn_frame, text="Save Figure", command=self._save_fig, state=tk.DISABLED)
        self.btn_savefig.pack(side=tk.LEFT, padx=4)

        # Progress bar + status
        pfrm = ttk.Frame(self, padding=(12, 0, 12, 8))
        pfrm.pack(side=tk.TOP, fill=tk.X)

        self.pb = ttk.Progressbar(pfrm, mode="determinate", maximum=100)
        self.pb.pack(fill=tk.X, pady=6)

        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(pfrm, textvariable=self.var_status).pack(side=tk.LEFT)

    def _build_results(self):
        frm = ttk.LabelFrame(self, text="Results", padding=12)
        frm.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 8))

        # Estimated stats
        self.var_mean = tk.StringVar(value="—")
        self.var_sem = tk.StringVar(value="—")
        self.var_ci = tk.StringVar(value="—")
        self.var_n = tk.StringVar(value="—")
        self.var_time = tk.StringVar(value="—")

        grid = ttk.Frame(frm)
        grid.pack(side=tk.LEFT, fill=tk.X, expand=True)

        r = 0
        ttk.Label(grid, text="Estimated mean E[D]:").grid(row=r, column=0, sticky="w")
        ttk.Label(grid, textvariable=self.var_mean, font=("TkDefaultFont", 10, "bold")).grid(row=r, column=1, sticky="w", padx=8)

        r += 1
        ttk.Label(grid, text="Standard error (SEM):").grid(row=r, column=0, sticky="w")
        ttk.Label(grid, textvariable=self.var_sem).grid(row=r, column=1, sticky="w", padx=8)

        r += 1
        ttk.Label(grid, text="95% CI:").grid(row=r, column=0, sticky="w")
        ttk.Label(grid, textvariable=self.var_ci).grid(row=r, column=1, sticky="w", padx=8)

        r += 1
        ttk.Label(grid, text="Samples used:").grid(row=r, column=0, sticky="w")
        ttk.Label(grid, textvariable=self.var_n).grid(row=r, column=1, sticky="w", padx=8)

        r += 1
        ttk.Label(grid, text="Elapsed time:").grid(row=r, column=0, sticky="w")
        ttk.Label(grid, textvariable=self.var_time).grid(row=r, column=1, sticky="w", padx=8)

        # Comparison block
        cmpf = ttk.LabelFrame(frm, text="Analytic reference (if available)", padding=8)
        cmpf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.var_n_label = tk.StringVar(value="n = —")
        ttk.Label(cmpf, textvariable=self.var_n_label, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.var_ref = tk.StringVar(value="—")
        ttk.Label(cmpf, text="Known E[D]:").pack(anchor="w")
        ttk.Label(cmpf, textvariable=self.var_ref).pack(anchor="w")

        self.var_diff = tk.StringVar(value="—")
        ttk.Label(cmpf, text="Difference (est - ref):").pack(anchor="w")
        ttk.Label(cmpf, textvariable=self.var_diff).pack(anchor="w")

    def _build_plot(self):
        frm = ttk.LabelFrame(self, text="Histogram of sampled distances", padding=8)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Distance")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title("—")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frm)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    # ---------- Actions ----------
    def _run(self):
        try:
            dim = int(self.var_dim.get())
            total = int(self.var_samples.get())
            batch = int(self.var_batch.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid integers for n, Samples N, and Batch size.")
            return

        if dim < 1 or total < 1 or batch < 1:
            messagebox.showerror("Invalid input", "All parameters must be positive integers.")
            return

        seed_str = self.var_seed.get().strip()
        if seed_str and not seed_str.isdigit():
            if not messagebox.askyesno("Seed warning", "Seed is not an integer. Use as string anyway?"):
                return

        # Prepare UI
        self.pb["value"] = 0
        self.var_status.set("Running…")
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.DISABLED)
        self.btn_savefig.config(state=tk.DISABLED)

        self.var_mean.set("—")
        self.var_sem.set("—")
        self.var_ci.set("—")
        self.var_n.set("—")
        self.var_time.set("—")
        self.var_n_label.set(f"n = {dim}")
        self.var_ref.set("—")
        self.var_diff.set("—")

        self.ax.clear()
        self.ax.set_xlabel("Distance")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title("Sampling…")
        self.canvas.draw_idle()

        # Start worker
        self.cancel_event.clear()
        self.worker = SimWorker(dim, total, batch, seed_str, self.q, self.cancel_event)
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            self.var_status.set("Cancelling…")

    def _export_csv(self):
        if not hasattr(self, "_last_summary"):
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        s = self._last_summary
        with open(path, "w", encoding="utf-8") as f:
            f.write("n,samples,mean,std,sem,elapsed\n")
            f.write(f"{s['n_dim']},{s['n']},{s['mean']},{s['std']},{s['sem']},{s['elapsed']}\n")
        messagebox.showinfo("Export", "CSV saved.")

    def _save_fig(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not path:
            return
        self.fig.savefig(path, bbox_inches="tight")
        messagebox.showinfo("Save Figure", "Figure saved.")

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg["type"] == "progress":
                    pct = 100.0 * msg["processed"] / msg["total"]
                    self.pb["value"] = pct
                    eta = "—" if not np.isfinite(msg["eta"]) else str(timedelta(seconds=int(msg["eta"])))
                    self.var_status.set(f"Progress: {msg['processed']}/{msg['total']}  |  ETA: {eta}")
                    if msg["mean"] == msg["mean"]:  # not NaN
                        self.var_mean.set(f"{msg['mean']:.6f}")
                    if msg["sem"] == msg["sem"]:
                        self.var_sem.set(f"{msg['sem']:.6f}")


                elif msg["type"] == "done":

                    self.btn_run.config(state=tk.NORMAL)

                    self.btn_cancel.config(state=tk.DISABLED)

                    self.btn_export.config(state=tk.NORMAL)

                    self.btn_savefig.config(state=tk.NORMAL)

                    if msg["canceled"]:

                        self.var_status.set("Cancelled.")

                    else:

                        self.var_status.set("Done.")

                    s = msg["stats"]

                    mean_est = s["mean"]

                    self.var_mean.set(f"{mean_est:.6f}")

                    self.var_n.set(str(s["n"]))

                    self.var_time.set(str(timedelta(seconds=int(msg["elapsed"]))))

                    if s["sem"] == s["sem"]:

                        ci_low = s["mean"] - 1.96 * s["sem"]

                        ci_high = s["mean"] + 1.96 * s["sem"]

                        self.var_sem.set(f"{s['sem']:.6f}")

                        self.var_ci.set(f"[{ci_low:.6f}, {ci_high:.6f}]")

                    else:

                        self.var_sem.set("—")

                        self.var_ci.set("—")

                    # Analytic reference for n=1,2

                    n_dim = int(self.var_dim.get())

                    ref = None

                    if n_dim == 1:

                        ref = analytic_mean_n1()

                    elif n_dim == 2:

                        ref = analytic_mean_n2()

                    if ref is not None:

                        self.var_ref.set(f"{ref:.6f}")

                        self.var_diff.set(f"{mean_est - ref:+.6f}")

                    else:

                        self.var_ref.set("Not available")

                        self.var_diff.set("—")

                    # -----------------------------

                    # Plot histogram with mean line

                    # -----------------------------

                    hist = msg.get("hist", np.array([]))

                    self.ax.clear()

                    if hist.size > 0:

                        self.ax.hist(hist, bins=60, color="lightgray", edgecolor="black", alpha=0.8)

                        # --- ADDED: vertical lines for means ---

                        self.ax.axvline(mean_est, color="red", linestyle="--", linewidth=2,

                                        label=f"Monte Carlo mean = {mean_est:.5f}")

                        if ref is not None:
                            self.ax.axvline(ref, color="blue", linestyle=":", linewidth=2,

                                            label=f"Analytic mean = {ref:.5f}")

                        self.ax.legend(loc="upper right", fontsize=9)

                        # --------------------------------------

                        self.ax.set_title(f"Histogram of sampled distances (subsample n={hist.size})")

                    else:

                        self.ax.set_title("No samples collected for histogram")

                    self.ax.set_xlabel("Distance")

                    self.ax.set_ylabel("Frequency")

                    self.canvas.draw_idle()

                    # Stash for export
                    self._last_summary = {
                        "n_dim": n_dim,
                        "n": s["n"],
                        "mean": s["mean"],
                        "std": s["std"],
                        "sem": s["sem"],
                        "elapsed": msg["elapsed"],
                    }

                elif msg["type"] == "error":
                    self.btn_run.config(state=tk.NORMAL)
                    self.btn_cancel.config(state=tk.DISABLED)
                    messagebox.showerror("Error", msg["message"])
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)


if __name__ == "__main__":
    app = App()
    app.mainloop()
