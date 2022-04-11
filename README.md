# VPRobust
me/uzerbinati/UZBox/Maths/Numerica/KAUST/Mixed/VPRobust/ArchimedesKitten/HT_VelocityNorm_0.2_1.ppm' 

### Elements

HT: Hood-Taylor element with P2-P1 (velocity-pressure).

BP: Bercovier-Pironneau P1isoP2-P1 (velocity-pressure).

BP+: Discontinuous Bercovier-Pironneau P1isoP2-(P1+P0) (velocity-pressure).

MINI: (P1+B3)-P1 (velocity-pressure).

### Meshes

ST. Structured mesh of simplices
SCC. Structured criss-cross mesh
US. UnStructured mesh of simplices

| Test Name        | Description                                           | Elements   | Mesh      |
| ---------------- | ----------------------------------------------------- | ---------- | --------- |
| ArchimedesKitten | no flow example                                       | HT/BP/MINI | US        |
| GaussEland       | large velocity w.r.t. pressure                        | HT/BP/MINI | US        |
| CantorChameleon  | discrete div. w.r.t. increasing velocity              | HT/BP/MINI | US        |
| WeilBull         | ArchimedesKitten with grad-div stabil. fixed mesh     | HT/BP/MINI | US,ST,SCC |
| NashWolf         | GaussEland with grad-div stabil. fixed mesh           | HT/BP/MINI | US,ST,SCC |
| BolyaiBadger     | CantorChameleon with grad-div stabil. fixed mesh      | HT/BP/MINI | US,ST,SCC |

