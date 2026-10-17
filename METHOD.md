# Theory and optimization method

This project treats cosmological parameter fitting as a numerical optimization problem. The physical model supplies a predicted expansion history, and particle swarm optimization (PSO) searches the model parameter space for the minimum of the observational mismatch.

## Cosmological background

For a homogeneous and isotropic universe, the Friedmann--Lemaître--Robertson--Walker line element can be written as

$$
ds^2=-c^2dt^2+a^2(t)\left[\frac{dr^2}{1-Kr^2}+r^2d\Omega^2\right].
$$

The Friedmann equation relates the expansion rate to the total energy density and spatial curvature. Written in terms of present-day density parameters, the dimensionless expansion rate $E(z)=H(z)/H_0$ used in this project is

$$
E^2(z)=\Omega_m(1+z)^3+\Omega_k(1+z)^2+\Omega_X(1+z)^{3(1+w)},
$$

with

$$
\Omega_m+\Omega_k+\Omega_X=1.
$$

This form follows from the continuity equation for a component with a constant equation-of-state parameter $w$,

$$
\dot\rho+3H\left(\rho+\frac{p}{c^2}\right)=0,
\qquad p=w\rho c^2,
$$

which gives $\rho\propto(1+z)^{3(1+w)}$.

The models used in the repository are therefore:

- **Flat $\Lambda$CDM**: $\Omega_k=0$, $w=-1$, parameters $(H_0,\Omega_m)$.
- **Non-flat $\Lambda$CDM**: $w=-1$, parameters $(H_0,\Omega_m,\Omega_k)$.
- **Flat XCDM**: $\Omega_k=0$, constant $w$ is fitted, parameters $(H_0,\Omega_m,w)$.
- **Non-flat XCDM**: parameters $(H_0,\Omega_m,\Omega_k,w)$.

Here XCDM is a constant-$w$ phenomenological extension; the code does not fit a time-dependent $w(z)$.

## Cosmic-chronometer objective

For observed cosmic-chronometer measurements $(z_i,H_i,\sigma_i)$, a candidate parameter vector $\boldsymbol\theta$ is scored with

$$
\chi^2(\boldsymbol\theta)=\sum_i
\frac{\left[H_{\rm model}(z_i;\boldsymbol\theta)-H_i\right]^2}{\sigma_i^2}.
$$

Low $\chi^2$ corresponds to a theoretical expansion history that follows the measured $H(z)$ points more closely. PSO is used only to minimize this objective; it is not a posterior sampler.

## Particle swarm optimization

A particle is a candidate cosmology. In flat $\Lambda$CDM,

$$
\mathbf{x}_j=(H_{0,j},\Omega_{m,j}).
$$

Each particle remembers its own best position, $\mathbf{p}_j$ (`pbest`), while the swarm remembers the best position found by any particle, $\mathbf{g}$ (`gbest`). The update is

$$
\mathbf{v}_j^{(t+1)}=
\omega\mathbf{v}_j^{(t)}+
 c_1\mathbf{r}_1\odot(\mathbf{p}_j-\mathbf{x}_j^{(t)})+
 c_2\mathbf{r}_2\odot(\mathbf{g}-\mathbf{x}_j^{(t)}),
$$

followed by

$$
\mathbf{x}_j^{(t+1)}=\mathbf{x}_j^{(t)}+\mathbf{v}_j^{(t+1)}.
$$

The three terms represent inertia, attraction toward a particle's own best-known position, and attraction toward the swarm's global best. The random vectors $\mathbf{r}_1$ and $\mathbf{r}_2$ are sampled component-wise on $[0,1]$.

The production calculations use $\omega=0.72$ and $c_1=c_2=1.49$.

## A five-particle hand example

The presentation used five trial cosmologies to make the update mechanism concrete. Their initial scores are computed with the same 31-point $\chi^2$ function as the production run:

| Particle | $(H_0,\Omega_m)$ | $\chi^2$ |
|---|---:|---:|
| $P_1$ | $(60,0.20)$ | 88.92 |
| $P_2$ | $(68,0.30)$ | 15.05 |
| $P_3$ | $(75,0.50)$ | 116.90 |
| $P_4$ | $(85,0.20)$ | 61.34 |
| $P_5$ | $(55,0.70)$ | 32.56 |

At initialization `pbest` is simply the particle's starting position, and $P_2$ is the initial `gbest`. For the illustrative first update of $P_1$,

$$
\mathbf{x}_1^{(0)}=(60,0.20),\qquad
\mathbf{v}_1^{(0)}=(2.00,0.020),\qquad
\mathbf{g}=(68,0.30),
$$

and, using the presentation's random factors $\mathbf{r}_2=(0.80,0.70)$ while the first-step cognitive term vanishes,

$$
\mathbf{v}_1^{(1)}
=0.72(2.00,0.020)
+1.49(0.80,0.70)\odot[(68,0.30)-(60,0.20)]
=(10.976,0.1187),
$$

so

$$
\mathbf{x}_1^{(1)}=(70.976,0.3187).
$$

Its objective improves from $\chi^2=88.92$ to $17.67$. The method notebook reproduces this example and then transitions to the full 100-particle production swarm.

## Diagnosing the optimization

The repository uses several complementary diagnostics:

1. **Personal-best and global-best pulls** show how individual memory and collective memory guide different particles.
2. **Iteration snapshots** show the swarm contracting from the initial search box into the low-$\chi^2$ valley.
3. **Global-best path** follows the sequence of best-known cosmologies as the optimizer improves.
4. **Best-$\chi^2$ convergence** verifies that the objective decreases and eventually stabilizes.
5. **Parameter histories** show $H_0$ and $\Omega_m$ settling to the final best-fit values.

These diagnostics demonstrate optimizer behavior. They are not confidence contours or posterior samples.
