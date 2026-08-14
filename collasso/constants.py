"""
collasso.constants — costanti fisiche cgs per la simulazione del collasso
gravitazionale stellare (Step 2, esteso in Step 3).

Modulo senza dipendenze da altri moduli del progetto: solo variabili
modulo-livello, ciascuna con la propria fonte citata nel commento.
"""

from __future__ import annotations

# Costante di gravitazione universale (cm^3 g^-1 s^-2), CODATA 2018.
# NOTA (Step 2): non era utilizzata da `physical_profile`, che si basa solo
# sulla relazione massa-densita-geometria di Lane-Emden e non richiede G.
# Era stata inclusa per completezza della struttura concordata col planner e
# riservata all'uso negli step futuri.
# NOTA (Step 3): la riserva sopra e' ora chiusa — G_CGS e' finalmente
# utilizzata da `chandrasekhar_mass_msun` in `collasso/eos.py`.
G_CGS = 6.6743e-8

# Massa solare (g), da parametro solare nominale IAU 2015 (GM_sun/G).
M_SUN_G = 1.98892e33

# Fattore di conversione km -> cm (cm per km).
KM_CM = 1.0e5

# --- Costanti aggiunte in Step 3 (EOS di Chandrasekhar), CODATA 2018 -------

# Costante di Planck ridotta (erg*s), CODATA 2018.
HBAR_CGS = 1.054571817e-27

# Velocita' della luce nel vuoto (cm/s), CODATA 2018 (valore esatto per
# definizione del SI).
C_LIGHT_CGS = 2.99792458e10

# Massa dell'elettrone (g), CODATA 2018.
M_ELECTRON_G = 9.1093837015e-28

# Unita' di massa atomica unificata (g), CODATA 2018. NOTA: e' l'unita' di
# massa atomica (u), NON la massa del protone/idrogeno (m_H) — usata per
# convertire rho in densita' numerica di elettroni via n_e = Ye*rho/M_U_G.
M_U_G = 1.66053906660e-24

# --- Costante aggiunta in Step 6 (limite TOV, classificazione remnant) -----

# Massa del neutrone (g), CODATA 2018. Usata dall'EOS di neutroni degeneri
# (collasso/eos_neutron.py, gas di neutroni liberi non interagenti,
# formalismo di Oppenheimer & Volkoff 1939) al posto della massa
# dell'elettrone: n_n = rho/M_NEUTRON_G (nessun Ye, a differenza dell'EOS di
# elettroni degeneri di Step 3).
M_NEUTRON_G = 1.67492749804e-24
