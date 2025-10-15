# Maths-AA-HL-Second-IA---Average-Distance-Between-Two-Random-Points-in-an-n-Dimensional-Unit-Cube
Analytical and computational investigation of the expected distance between two random points in an n-dimensional unit cube.

# Maths AA HL Second IA — Average Distance Between Two Random Points in an *n*-Dimensional Unit Cube

This repository contains all the analytical derivations, LaTeX source files, Python code, and numerical verifications used in my **IB Mathematics: Analysis and Approaches HL Internal Assessment (IA)**.  
The investigation explores how the **expected Euclidean distance** between two points chosen uniformly at random within an *n*-dimensional unit hypercube \([0,1]^n\) behaves as the number of dimensions increases.

---

## Project Overview

The goal of this IA is to determine

\[
\mathbb{E}[D_n] = \text{Expected distance between two random points in } [0,1]^n
\]

where the Euclidean distance between two points  
\( A = (a_1, a_2, \ldots, a_n) \) and \( B = (b_1, b_2, \ldots, b_n) \) is given by

\[
D_n = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + \cdots + (a_n - b_n)^2}.
\]

The study begins with low-dimensional cases that can be solved analytically:

- **Case \(n = 1\):**
  \[
  \mathbb{E}[D_1] = \frac{1}{3}.
  \]

- **Case \(n = 2\):**
  Using integration in polar coordinates and probability density functions,  
  \[
  \mathbb{E}[D_2] = \frac{\sqrt{2} + 2 + 5\ln(1+\sqrt{2})}{15} \approx 0.5214.
  \]

These exact results serve as verification benchmarks for the computational model.  
For higher dimensions, a numerical Monte Carlo simulation is employed.

---

## Monte Carlo Simulation

A **Python Tkinter GUI** (`mc_distance_ui.py`) was developed to estimate  
\(\mathbb{E}[D_n]\) through large-scale random sampling.

### Features
- Works for **any dimension \(n \ge 1\)**.
- Adjustable number of samples, batch size, and random seed.
- Live **progress bar**, **ETA**, and **cancel button**.
- Displays **histogram of sampled distances** with:
  - Red dashed line → Monte Carlo mean  
  - Blue dotted line → Analytical mean (for \(n=1,2\))
- Exports results to CSV and saves plots as PNG.
- Uses **NumPy** for vectorised computation and **Matplotlib** for visualisation.

### Example Output
For \(n = 1\) with 200,000 samples:

\[
\text{Monte Carlo mean} = 0.33355 \quad\text{vs.}\quad \text{Analytical mean} = \tfrac{1}{3}.
\]

The distribution of sampled distances is shown below:

<img width="1029" height="368" alt="monte_carlo_plot_1" src="https://github.com/user-attachments/assets/a2c3e543-70ae-46d9-bdff-ad36fc166151" />


---

## Analytical Approach Summary

To derive the analytical expectations, the investigation uses:
- **Geometric probability** in \([0,1]^2\) to find the density function  
  \[
  f(x) = 2(1-x), \quad 0 \le x \le 1.
  \]
- **Transformation to polar coordinates**  
  \[
  x = r\cos\theta, \quad y = r\sin\theta, \quad dx\,dy = r\,dr\,d\theta.
  \]
- **Piecewise limits** for the unit square boundary:  
  \[
  r_{\max}(\theta) =
  \begin{cases}
  1 / \cos\theta, & 0 \le \theta \le \pi/4, \\
  1 / \sin\theta, & \pi/4 \le \theta \le \pi/2.
  \end{cases}
  \]
- Evaluated integrals:
  \[
  I_1 = \frac{1}{3}\!\left(\sqrt{2} + \ln(1+\sqrt{2})\right), \quad
  I_2 = \frac{1}{24}\!\left(22\sqrt{2} + 3\ln(1+\sqrt{2}) - 8\right), \quad
  I_3 = \frac{1}{15}\!\left(4 + 5\sqrt{2}\right).
  \]
- Final expression:
  \[
  \mathbb{E}[D_2] = 4(I_1 - 2I_2 + I_3)
  = \frac{\sqrt{2} + 2 + 5\ln(1+\sqrt{2})}{15}.
  \]

---

## 🧩 Repository Structure

```
Maths-AA-HL-Second-IA/
├── README.md                # Project overview and methodology
├── Maths_IA_Final.pdf       # Compiled final IA document
├── mc_distance_ui.py        # Monte Carlo simulation app (Tkinter GUI)
└── images/                  # Plots, figures, and verification screenshots
```


---

## Technologies Used
- **Python 3.10+**
  - `numpy` — fast vectorised computation  
  - `matplotlib` — visualisation  
  - `tkinter` — GUI  
- **LaTeX (PGFPlots, TikZ)** — for analytical figures and surface plots  

---

## Key Insights
- The expected distance **increases with dimension** — as \(n\) grows, points become more “spread out” within the hypercube.  
- Monte Carlo results converge to analytical values for low \(n\), validating the accuracy of the simulation algorithm.  
- The combination of **calculus, probability, and computational methods** highlights the interplay between analytic reasoning and numerical experimentation in higher-dimensional mathematics.

---

## Acknowledgements
- Project developed as part of the **IB Mathematics: Analysis and Approaches HL** curriculum at *Southbank International School (2025)*.  
- Analytical references:  
  - Weisstein, E. W. “Line Picking.” *MathWorld–Wolfram*  
  - University College London, *Polar Coordinates Tutorial* (2024)  

---

## License
This project is shared under the **MIT License** for educational and non-commercial use.  
Feel free to fork, reference, or adapt the simulation for further mathematical exploration.

