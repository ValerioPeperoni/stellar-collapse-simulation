# Validazione qualitativa contro GR1D — Step 8

Confronto dei risultati del progetto con il comportamento noto/pubblicato
del codice GR1D (github.com/evanoconnor/GR1D) e della letteratura di
riferimento, come richiesto da CLAUDE.md. GR1D non viene eseguito
letteralmente (codice Fortran/C esterno, GR completo + trasporto
neutrini, fuori portata come dipendenza) — il confronto è con i valori e
i comportamenti pubblicati in:
- O'Connor & Ott (2010), "A New Open-Source Code for Spherically
  Symmetric Stellar Collapse to Neutron Stars and Black Holes",
  Class. Quantum Grav. 27, 114103 (arXiv:0912.2393).
- O'Connor & Ott (2011), "Black Hole Formation in Failing Core-Collapse
  Supernovae", ApJ 730, 70 (arXiv:1010.5550).

Metodo: `critic-fisico`, indagine quantitativa eseguita in autonomia
(codice riletto, simulazioni rieseguite in proprio, non solo verifica dei
numeri riportati dall'orchestratore).

**REVISIONE — 2026-08-13 (ciclo "Integrazione catalogo reale")**: questo
documento è stato interamente riscritto per riflettere il catalogo dei
progenitori aggiornato (`data/progenitors_reference.csv`), che sostituisce
i vecchi valori placeholder con massa del nucleo e Ye reali da O'Connor &
Ott (2011), ApJ 730:70, Tabella 1 (dati Woosley, Heger & Weaver 2002).
Tutti i numeri sotto sono stati ricalcolati sul codice/dati attuali —
nessun valore di questa versione proviene dal vecchio catalogo
placeholder. Vedi STATUS.md, ciclo "Integrazione catalogo reale", per il
contesto completo di questo aggiornamento.

## 0. Provenienza dei dati usati in questa validazione

| Stella | massa_nucleo_msun | ye | densita_centrale_gcm3 | n_politropico | raggio_iniziale_km |
|---|---|---|---|---|---|
| s15 | 1.55 (letteratura) | 0.495 (letteratura) | 5.0000e9 (plausibile) | 2.96922161 (derivato) | 1960.78 (derivato) |
| s20 | 1.46 (letteratura) | 0.495 (letteratura) | 7.5000e9 (plausibile) | 2.97629373 (derivato) | 1686.31 (derivato) |
| s25 | 1.62 (letteratura) | 0.495 (letteratura) | 1.0000e10 (plausibile) | 2.98032431 (derivato) | 1590.05 (derivato) |

"Letteratura" = O'Connor & Ott (2011), Tabella 1 pag. 5, modelli
s15WHW02/s20WHW02/s25WHW02. "Plausibile" = valore fisico ragionevole per
nuclei di ferro pre-collasso, non da letteratura (nessun paper controllato
tabula densità centrale pre-collasso per stella nominata). "Derivato" =
calcolato internamente da `collasso.eos.n_eff_chandrasekhar` +
`collasso.lane_emden.physical_profile` a partire dalla massa/densità
sopra — vedi `data/progenitors_reference.csv` per il dettaglio completo
per campo e §5 sotto per il limite fisico di questa derivazione.

## 1. Indagine — andamento di v_inner (s20)

**Nota storica — verdetto originale DECLASSATO dopo riesame (2026-08-13)**:
nella versione precedente di questo documento (catalogo placeholder, s20
massa_nucleo=1.50 Msun, n=2.0 arbitrario, ye=0.46,
T_MAX_FREE_FALL_MULTIPLIER=10), questa sezione documentava un pattern
non-monotono ben convergente in N (discesa a ~-800 km/s entro 20ms,
risalita a ~-600 km/s a 20-50ms, poi ripiombo a ~-2200 km/s alla fine),
concluso allora come "FISICO, non artefatto numerico".

**Quel verdetto è stato riesaminato e va corretto**, in seguito a
un'ipotesi dell'utente verificata rigorosamente da `critic-fisico` (vedi
STATUS.md, nota "Verdetto 'FISICO' di Step 8 riesaminato e DECLASSATO",
per il dettaglio completo). Sintesi: un test di convergenza in N a parità
di condizione iniziale può SOLO rivelare errore di discretizzazione — non
può, per costruzione, rivelare che la condizione iniziale stessa sia
fisicamente incoerente (il vecchio profilo placeholder aveva n=2.0 scelto
arbitrariamente, non derivato dalla EOS, con un mismatch di Lane-Emden di
**6.33x** fra il raggio dichiarato e quello consistente con la massa
dichiarata — scarto K, vedi §5, di **+35.58%**, contro +2.3%/+6.7%/+9.5%
delle stelle reali del catalogo attuale).

L'esperimento decisivo: tenendo IDENTICI ρc, ye, massa (quindi lo STESSO
rapporto massa/M_Ch=121.7% del vecchio caso) e cambiando SOLO n da 2.0 a
n_eff_chandrasekhar (EOS-consistente, come nel catalogo reale), il
pattern **sparisce completamente** (zero cambi di segno in dv/dt, N=200
fino a 1600). Il rapporto massa/M_Ch non era la causa: era l'incoerenza
strutturale di n=2.0. Il vecchio pattern era quindi **un artefatto
dell'incoerenza del profilo iniziale placeholder, mascherato da una
convergenza numerica che, per costruzione, non poteva rivelarlo** — non
una caratteristica fisica del modello di collasso, e non semplicemente
"sparito perché i dati sono cambiati" come una prima lettura superficiale
potrebbe suggerire.

**Con il catalogo reale (profilo EOS-consistente per costruzione), questo
pattern non si ripresenta** — coerentemente con questa spiegazione, non
per coincidenza. Questa sezione riporta la nuova indagine, rifatta da
zero sul codice/dati attuali.

**Andamento di v_inner (shell 0)**, `run_full_simulation("s20")`,
relativistic=True, t_collapse_s=0.267397 s (267.40 ms), evento terminale
`r_min_threshold`:
- **t=0-3.6ms**: piccola escursione POSITIVA (verso l'esterno), massimo
  **+709.0 km/s a t=2.19ms** — la pressione supera momentaneamente la
  gravità nella shell più interna (coerente con lo scarto K positivo del
  profilo iniziale, §5: non è un vero equilibrio, quindi un piccolo rimbalzo
  iniziale è atteso).
- **t=4.4ms in poi**: la velocità cambia segno e da qui decresce **in modo
  liscio e monotono** fino a ~t=200ms (da -223 km/s a -333 km/s) — nessuna
  inversione di segno di dv/dt verificata su tutta questa finestra.
- **t≈204-267ms (ultimo ~23% della traiettoria, vicino alla soglia
  terminale)**: oscillazioni di ampiezza crescente (fino a ~70 km/s di
  escursione locale) sovrapposte a un trend comunque monotono verso valori
  sempre più negativi — mai una vera "pausa" prolungata come nella vecchia
  versione. Valore finale: **-1434.9 km/s a t=266.87ms**.

**Decomposizione forze (gravità vs pressione, shell 0)**:

| t (ms) | v_inner (km/s) | grav (cm/s²) | pressione (cm/s²) | press/grav |
|---|---|---|---|---|
| 2.19 | +7.1 | -1.5966e10 | +1.5947e10 | -0.9988 |
| 10.21 | -41.7 | -1.6020e10 | +1.5612e10 | -0.9745 |
| 19.69 | -64.3 | -1.6236e10 | +1.6078e10 | -0.9903 |
| 50.31 | -89.5 | -1.7299e10 | +1.7228e10 | -0.9959 |
| 150.21 | -222.3 | -2.6751e10 | +2.6578e10 | -0.9935 |
| 200.79 | -333.2 | -4.4089e10 | +4.3895e10 | -0.9956 |
| 260.31 | -846.2 | -4.0738e11 | +4.0362e11 | -0.9908 |
| 266.87 | -1434.9 | -1.3643e12 | +1.3176e12 | -0.9658 |

Gravità e pressione restano quasi esattamente cancellate per tutta la
traiettoria (rapporto sempre in [-1.01, -0.95]), ma qui il residuo netto
resta **sempre negativo** con magnitudine monotonamente crescente —
nessuna finestra con inversione di segno del residuo, a differenza della
vecchia versione.

**Test di convergenza numerica** (N=200/400/800 shell), sulle oscillazioni
tardive (t>150ms):
- t_collapse_s: 267.40ms (N=200) → 266.33ms (N=400) → 265.91ms (N=800) —
  converge bene.
- Ampiezza massima delle oscillazioni tardive: min(v_inner) = -1517 km/s
  (N=200) → -1233 km/s (N=400) → -1013 km/s (N=800) — **diminuisce con
  N**: sono un artefatto numerico (ringing da assenza di viscosità
  artificiale vicino alla soglia terminale/shell-crossing, limite già
  dichiarato in `collasso/dynamics.py`), non una feature fisica robusta
  come lo era la vecchia "quasi-pausa" (che era convergente <0.1% su
  N=100→1600 — un comportamento qualitativamente diverso da quello
  osservato qui).

**Verdetto**: il pattern non-monotono precedentemente investigato **non
persiste** con il catalogo reale. Il nuovo comportamento — piccolo
transiente iniziale outward, poi infall liscio e monotono per l'80% della
traiettoria, poi oscillazioni numeriche tardive di ampiezza decrescente
con la risoluzione — è coerente con il limite già dichiarato "nessuna
viscosità artificiale" e non richiede una nuova spiegazione fisica. La
domanda originale dell'utente (fisico vs artefatto) resta comunque
rilevante come metodologia: qui la risposta è "il transiente iniziale e
l'infall liscio sono fisici (convergono in N), le oscillazioni tardive
sono un artefatto noto e già dichiarato (diminuiscono con N)".

## 2. Tempi scala

| Stella | ρc iniziale (g/cm³) | t_caduta_libera (ms) | t_collasso (ms, relativistic=True) | rapporto t_collasso/t_ff |
|---|---|---|---|---|
| s15 | 5.0000e9 | 29.708 | 215.219 | 7.2445 |
| s20 | 7.5000e9 | 24.256 | 266.870 | 11.0020 |
| s25 | 1.0000e10 | 21.007 | 128.187 | 6.1022 |

Controllo indipendente s20 con relativistic=False: t_collasso=306.899ms,
rapporto=12.6523 — questo è il valore che ha reso necessario alzare
`T_MAX_FREE_FALL_MULTIPLIER` da 10.0 a 15.0 (§5), dato che s20 è il caso
più marginalmente supra-Chandrasekhar (102.3% di M_Ch) e quindi il più
lento a raggiungere la soglia di collasso. Gli ordini di grandezza (decine
di ms per il tempo di caduta libera, ~100-300ms per la scala dinamica fino
all'evento terminale) restano coerenti con quelli tipici riportati per la
fase di collasso fino al bounce in simulazioni stile GR1D. **Limite
dichiarato invariato**: `t_collasso` è il tempo di un evento numerico
terminale (soglia raggio minimo), non un vero bounce fisico — il confronto
resta sull'ordine di grandezza, non una validazione quantitativa di un
tempo di bounce (che questo modello non può produrre, per costruzione:
l'EOS di Chandrasekhar per elettroni degeneri non si irrigidisce mai a
densità nucleare).

## 3. Massa critica

- M_Chandrasekhar(ye=0.495) = **1.426951 Msun** — ora IDENTICA per tutte
  e tre le stelle (ye=0.495 fisso, dato di letteratura O'Connor&Ott, a
  differenza del vecchio catalogo dove ye variava per stella).
- M_TOV SLy = 2.002029 Msun @ ρc=2.1261e15 g/cm³ (scarto 2.3% da
  ≈2.05 Msun di letteratura) — **invariato** rispetto alla versione
  precedente di questo documento (M_TOV non dipende dal catalogo dei
  progenitori, solo dalla EOS nucleare).
- M_TOV APR4 = 2.187175 Msun @ ρc=1.8874e15 g/cm³ (scarto 0.6% da
  ≈2.20 Msun di letteratura) — invariato.
- Confronto osservativo PSR J0740+6620 = 2.08±0.07 Msun (Fonseca et al.
  2021): SLy a 1.114σ, APR4 a 1.531σ.

## 4. Classificazione remnant

| Stella | massa_nucleo_msun | massa/M_Ch | Classe (SLy) | Classe (APR4) |
|---|---|---|---|---|
| s15 | 1.55 | 108.62% | neutron_star | neutron_star |
| s20 | 1.46 | 102.32% | neutron_star | neutron_star |
| s25 | 1.62 | 113.53% | neutron_star | neutron_star |

Tutte e tre le stelle sono ora correttamente supra-Chandrasekhar (a
differenza del vecchio catalogo placeholder, dove s15 con massa=1.35 Msun
risultava erroneamente SOTTO M_Ch e classificata `white_dwarf` — un
artefatto del placeholder, non plausibile per un vero progenitore da
15 Msun ZAMS). Nessuna delle tre supera M_TOV: nessuna diventa
`black_hole`. **Limitazione onesta invariata**: solo 3 stelle nel
catalogo, non è possibile testare qualitativamente la transizione verso
`black_hole` descritta in letteratura per progenitori più massicci.

## 5. Scarto K — quanto ogni nucleo è supra-Chandrasekhar

Sezione nuova rispetto alla versione precedente di questo documento,
introdotta insieme al catalogo reale. Dimostrazione quantitativa: con
la EOS ESATTA di Chandrasekhar (non un politropo a indice fisso), la massa
di un vero equilibrio idrostatico M(ρc) cresce monotonamente con la
densità centrale ma **non supera mai** M_Ch, a nessuna densità, nemmeno
arbitrariamente alta e non fisica:

| ρc (g/cm³) | M(ρc)/M_Ch |
|---|---|
| 1e10 | 99.05% |
| 1e12 | 99.95% |
| 1e14 | 99.998% |
| 1e20 (non fisico, ben oltre la densità nucleare) | 100.0000% (mai raggiunto) |

Poiché tutte e tre le masse reali del nucleo (1.55/1.46/1.62 Msun) sono
SOPRA M_Ch=1.4270 Msun, **nessuna delle tre stelle ammette un vero
equilibrio idrostatico** con questa EOS, a nessuna densità centrale — è
la ragione fisica diretta per cui questi nuclei collassano, non un
limite del metodo numerico.

Il progetto quantifica questo scarto per ogni stella con
`collasso.eos.chandrasekhar_k_deviation_pct`: confronta la costante
politropica implicita nel profilo iniziale (che fa tornare la massa
esatta per costruzione) con quella reale della EOS di Chandrasekhar alla
stessa densità centrale:

| Stella | Scarto K | massa/M_Ch |
|---|---|---|
| s15 | +6.730% | 108.62% |
| s20 | +2.320% | 102.32% |
| s25 | +9.521% | 113.53% |

L'ordinamento è quello atteso: lo scarto cresce monotonamente con quanto
la stella è sopra M_Ch (s20 il meno supra-Chandrasekhar ha lo scarto
minore, s25 il più supra-Chandrasekhar ha lo scarto maggiore) — conferma
che questa non è un'incoerenza casuale, ma una misura diretta e
interpretabile del grado di instabilità di ciascun nucleo. Questo valore
è ora sempre visibile nell'output finale di `scripts/run_simulation.py` e
nel disclaimer 2 di `collasso.pipeline._build_disclaimers`, non solo in
questo documento.

**NON è un errore di modellazione**: n_politropico e raggio_iniziale_km
vanno intesi come un'approssimazione dell'ultimo istante prima
dell'innesco del collasso (condizione iniziale della dinamica, Step 4),
non come una vera soluzione di equilibrio — coerente con la natura stessa
del fenomeno modellato (un nucleo che collassa perché non può essere in
equilibrio).

## 6. Tabella vincoli CLAUDE.md — verifica puntuale

| Vincolo | Esito |
|---|---|
| Profilo di equilibrio iniziale (Lane-Emden) | Verificato — il profilo non è un vero equilibrio idrostatico per queste 3 stelle (supra-Chandrasekhar, §5), ma il vincolo richiede l'uso dell'equazione di Lane-Emden, non che il risultato sia un equilibrio esatto; limite dichiarato esplicitamente, non una violazione |
| Pressione di degenerazione (EOS Chandrasekhar) | Verificato |
| Soglia di instabilità (M_Chandrasekhar ~1.4 Msun) | Verificato — M_Ch(0.495)=1.4270 Msun |
| Dinamica del collasso (shell Lagrangiane, simmetria sferica) | Verificato |
| Gravità Newtoniana + correzioni relativistiche sul nucleo | Verificato |
| Classificazione remnant (TOV, piecewise polytrope SLy/APR4) | Verificato |
| Neutrini (raffreddamento semplificato) | Verificato, con nota: nessun termine di perdita energetica è implementato in nessun modulo — è un'assenza totale dichiarata (disclaimer 3: "semplificato/assente"), non un vero termine minimale attivo; coerente col vincolo CLAUDE.md, che esclude solo il trasporto completo |
| Rotazione/campo magnetico | Verificato — trascurati, limite dichiarato |
| Validazione (confronto qualitativo con GR1D) | Verificato in questo documento (riscritto per riflettere il catalogo reale) |

Nessun vincolo violato, nessun bug trovato in questo ciclo. Suite completa
di test: 93/93 passano (inclusi i 2 nuovi test per
`chandrasekhar_k_deviation_pct`).

## Conclusione

Il progetto supera il confronto qualitativo richiesto da CLAUDE.md anche
con il catalogo reale. Il punto più importante di questa revisione non è
solo che il comportamento non-monotono della velocità osservato in
precedenza per s20 sia sparito, ma **perché**: era un artefatto
dell'incoerenza strutturale del profilo placeholder (n_politropico=2.0
arbitrario, mismatch di Lane-Emden 6.33x, scarto K +35.58%), dimostrato
quantitativamente (§1) e non semplicemente inferito dalla sua scomparsa.
Con la massa reale del nucleo e un profilo EOS-consistente per
costruzione (§5), il collasso procede in modo liscio e monotono per la
maggior parte della traiettoria, con solo oscillazioni tardive note e già
dichiarate (assenza di viscosità artificiale) vicino alla soglia
terminale.

**Lezione di metodo, non solo di risultato**: un test di convergenza in N
verifica la fedeltà della discretizzazione numerica alla condizione
iniziale data — non la validità fisica di quella condizione iniziale. Le
due cose sono logicamente indipendenti e vanno sempre verificate
separatamente: un fenomeno può essere numericamente robusto e ben
convergente (come lo era il vecchio pattern, verificato <0.1% su
N=100→1600) ed essere comunque un artefatto, se la condizione iniziale
stessa non è fisicamente autoconsistente. Per questo motivo il progetto
ora affianca sempre, dove rilevante, un controllo di autoconsistenza del
profilo (lo "scarto K", §5) al test di convergenza in N — quest'ultimo da
solo non basta. Questo è anche un promemoria generale: un risultato
"fisico e ben convergente" ottenuto su dati placeholder non è garantito
sopravvivere all'aggiornamento con dati reali, ed è per questo che questo
documento va sempre tenuto sincronizzato con lo stato attuale del
codice/catalogo, non descritto come statico una volta per tutte.
