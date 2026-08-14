# Stato del progetto — aggiornato ad ogni ciclo

## Step
- [x] Step 1: Setup ambiente, import catalogo progenitori, parsing parametri stella scelta
- [x] Step 2: Solver Lane-Emden, profilo di equilibrio iniziale (verificare contro soluzioni analitiche note per n=0,1,5)
- [x] Step 3: EOS di Chandrasekhar (non-rel. -> ultra-rel.)
- [x] Step 4: Dinamica shell Lagrangiane, loop temporale del collasso
- [x] Step 5: Correzioni relativistiche sul nucleo
- [x] Step 6: Check limite TOV, classificazione remnant
- [x] Step 7: Pipeline di visualizzazione (Matplotlib: animazione + grafici numerici)
- [x] Step 8: Validazione qualitativa contro GR1D
- [~] Step 9: Pulizia, documentazione, README fatti; pubblicazione GitHub in attesa di conferma esplicita dell'utente

## Piano ciclo corrente (Step 1) — REV. 2 (corretto dopo review)

Decisioni vincolanti già prese con l'utente (non rimetterle in discussione):
Python 3.12 via winget (installazione manuale, fuori dal compito del coder),
venv `.venv` in root con requirements.txt (numpy, scipy, matplotlib, pandas,
pytest), catalogo iniziale = 3 stelle campione hardcoded (15/20/25 Msun) da
letteratura Sukhbold/Woosley & Heger, dichiarate esplicitamente come
placeholder in attesa del catalogo completo.

Nota di scope: il planner (questo documento) fissa ORA numericamente i range
fisici plausibili per Ye e n (vedi sotto) e i criteri di plausibilità rispetto
alla massa di Chandrasekhar. Il coder deve implementare la validazione hard
contro questi range nel dataclass e nei test — non si demanda più questa
responsabilità al critic-fisico a posteriori; il critic-fisico resta
comunque il secondo controllo indipendente prima della chiusura del ciclo.

### Range fisici vincolanti per questo step (fissati dal planner)
- **Ye (frazione di elettroni)**: range plausibile per nuclei di
  ferro/silicio pre-collasso = **0.42 – 0.50**. Valori fuori da questo
  range per il catalogo di riferimento sono da considerarsi errore di
  inserimento dati, non variabilità fisica accettabile in questo step.
- **n (indice politropico)**: range plausibile nel contesto EOS degenere
  = **1.5 – 3.0** (n=3/2 caso non-relativistico, n=3 caso ultra-relativistico;
  userà il solver di Lane-Emden allo Step 2).
- **`n_politropico` — limite di modello dichiarato (stile CLAUDE.md,
  analogo a neutrini/rotazione)**: per lo Step 1, `n` è un singolo valore
  fisso per stella, usato come approssimazione iniziale per l'equilibrio
  di Lane-Emden. Step 1 NON implementa la transizione EOS non-rel. ->
  ultra-rel. a indice variabile: quella è responsabilità esclusiva dello
  Step 3 (EOS di Chandrasekhar). Il codice e i commenti di Step 1 devono
  dichiarare esplicitamente questa semplificazione, non presentarla come
  EOS completa.
- **`raggio_iniziale_km` — solo letteratura, mai derivato in Step 1**: per
  questo step il valore deve provenire SEMPRE da letteratura/placeholder
  dichiarato in `fonte`. È vietato derivarlo ad hoc (es. da densita
  centrale + massa con formule improvvisate). La derivazione rigorosa del
  profilo di raggio tramite equazione di Lane-Emden è responsabilità
  esclusiva dello Step 2, che userà questo valore come cross-check, non
  come fonte primaria.
- **Massa di Chandrasekhar — criterio di plausibilità operativo**:
  `massa_nucleo_msun` per i progenitori di riferimento deve ricadere nel
  range **1.3 – 2.0 Msun**, coerente con l'ordine di grandezza della
  soglia di Chandrasekhar (~1.4 Msun, da tabella vincoli CLAUDE.md) che
  giustifica fisicamente l'instabilità e il collasso. Questo è un
  criterio operativo esplicito da tradurre in test, non un controllo
  generico lasciato al critic-fisico.

### 1. Setup ambiente
1.1. Verificare che Python 3.12 sia già installato (`python --version` /
     `py -3.12 --version`); se assente, FERMARSI e segnalare all'utente —
     non tentare l'installazione (compito già assegnato fuori da questo step).
1.2. Creare virtualenv in root: `python -m venv .venv`.
1.3. Creare `requirements.txt` in root con: numpy, scipy, matplotlib, pandas,
     pytest (versioni recenti compatibili con Python 3.12, non pinnate a
     patch-level salvo necessità).
1.4. Attivare `.venv` e installare le dipendenze (`pip install -r requirements.txt`).
1.5. Verifica rapida: `pip list` / import di numpy, scipy, matplotlib,
     pandas, pytest senza errori. Output filtrato: riportare solo
     esito ok/errore, non il log completo di pip.
1.6. (Facoltativo, non bloccante per questo step) aggiungere un
     `.gitignore` con `.venv/`, `__pycache__/`, `*.pyc` — utile in vista
     della futura pubblicazione GitHub (Step 9), ma il repo non è ancora
     inizializzato come git repo e non è richiesto inizializzarlo ora.

### 2. Package `collasso/`
2.1. Creare `collasso/__init__.py` (package minimale, eventualmente con
     `__version__`).
2.2. Creare `collasso/catalog.py` con:
     - dataclass `Progenitor` con campi:
       - `id: str` (identificativo stella, es. "s15", "s20", "s25")
       - `massa_zams_msun: float` (massa iniziale ZAMS del progenitore,
         cioè la massa "di targa" della stella, es. 15/20/25 Msun — NON
         confondere con la massa del nucleo pre-collasso; rinominato da
         `massa_msun` per evitare ambiguità). Nota per il futuro: il
         catalogo completo Sukhbold/WH avrà probabilmente anche una massa
         presupernova distinta dalla ZAMS; per lo Step 1, con il set di
         riferimento minimo a 3 stelle, questi due campi (ZAMS + nucleo)
         sono sufficienti e non si introduce un terzo campo ora.
       - `massa_nucleo_msun: float` (massa del nucleo pre-collasso,
         parametro chiave, userà Step 2/3; deve rispettare il criterio di
         plausibilità Chandrasekhar sopra, 1.3–2.0 Msun)
       - `densita_centrale_gcm3: float` (parametro chiave per Lane-Emden, Step 2)
       - `raggio_iniziale_km: float` (SEMPRE da letteratura/placeholder,
         mai derivato in questo step — vedi nota sopra; userà Step 2 come
         cross-check dopo il calcolo Lane-Emden)
       - `ye: float` (frazione di elettroni, range vincolante 0.42–0.50)
       - `n_politropico: float` (indice politropico, range vincolante
         1.5–3.0; valore fisso per stella, limite di modello dichiarato —
         vedi nota sopra)
       - `fonte: str` (riferimento bibliografico esplicito, es.
         "Sukhbold, Woosley & Heger 2016 — valore placeholder"; deve
         contenere la dicitura "placeholder" per il catalogo di
         riferimento — vedi test 5.7)
     - Validazione in `__post_init__` (hard, contro i range fisici fissati
       sopra, non generica):
       - `0.42 <= ye <= 0.50` (non il generico `0 <= ye <= 1`);
       - `1.5 <= n_politropico <= 3.0`;
       - `1.3 <= massa_nucleo_msun <= 2.0` (criterio di plausibilità
         Chandrasekhar);
       - masse e densità/raggio positivi;
       - `massa_nucleo_msun <= massa_zams_msun`;
       - sollevare `ValueError` con messaggio esplicito (che indichi il
         range atteso) in caso di violazione di uno qualunque di questi
         vincoli.
     - `load_catalog_from_csv(path) -> list[Progenitor]`: parsing generico
       ed estensibile, pensato per poter leggere in futuro il catalogo
       completo Sukhbold/WH senza riscrivere la funzione. Deve gestire
       esplicitamente il commento bibliografico in testa al file CSV:
       saltare le righe che iniziano con `#` (es. `comment='#'` se si usa
       `pandas.read_csv`, oppure skip manuale se si usa il modulo `csv`).
     - `load_reference_catalog() -> list[Progenitor]`: wrapper che carica
       `data/progenitors_reference.csv` (il set dei 3 placeholder).
     - `get_progenitor_by_id(catalog, id) -> Progenitor`: helper di
       selezione; solleva errore chiaro se l'id non esiste.

### 3. Dati
3.1. Creare `data/progenitors_reference.csv` con 3 righe (15, 20, 25 Msun),
     colonne coerenti con i campi della dataclass `Progenitor` (incluso
     `massa_zams_msun` rinominato).
3.2. Includere in testa al file una o più righe di commento (`#`) con la
     fonte bibliografica (Sukhbold, Woosley & Heger) e la dicitura
     esplicita "placeholder — in attesa del catalogo completo"; il parser
     CSV (2.2) deve saltare correttamente queste righe.
3.3. I valori numerici devono rispettare i range vincolanti fissati sopra
     da questo piano (Ye 0.42–0.50, n 1.5–3.0, massa nucleo 1.3–2.0 Msun)
     — non solo "plausibili rispetto alla tabella vincoli", ma
     numericamente dentro questi range specifici. Il critic-fisico
     effettua comunque un secondo controllo indipendente prima di
     chiudere il ciclo (vedi 6.3).

### 4. Script demo
4.1. Creare `scripts/step1_demo.py`:
     - carica il catalogo di riferimento,
     - stampa una tabella riassuntiva compatta delle 3 stelle (una riga
       per stella, colonne principali, inclusi `massa_zams_msun` e
       `massa_nucleo_msun` come campi distinti),
     - seleziona una "stella scelta" (proposta: 20 Msun, caso intermedio)
       da usare come default per gli step successivi,
     - stampa i parametri estratti della stella scelta in formato leggibile.
4.2. Nessun log grezzo o verboso in output: solo la tabella e il riepilogo
     finale (regola CLAUDE.md su filtraggio dell'output).

### 5. Test (`tests/test_catalog.py`)
5.1. Test caricamento catalogo di riferimento: 3 stelle, nessuna eccezione.
5.2. Test range fisico Ye: per ogni stella del catalogo di riferimento,
     verificare **con i valori concreti letti dal CSV** che
     `0.42 <= ye <= 0.50` (non un test tautologico che ricontrolla solo
     la logica del dataclass — deve leggere i dati reali del catalogo).
5.3. Test range fisico n: analogamente, verificare con i valori concreti
     del catalogo che `1.5 <= n_politropico <= 3.0`.
5.4. Test masse: positive, `massa_nucleo_msun <= massa_zams_msun` per ogni
     stella, e **test esplicito del criterio di plausibilità
     Chandrasekhar**: `1.3 <= massa_nucleo_msun <= 2.0` per ogni stella
     del catalogo di riferimento (non generico "massa positiva").
5.5. Test validazione dataclass (casi negativi, ai limiti dei range):
     costruire `Progenitor` con `ye` fuori range (es. 0.6 e 0.3, non solo
     1.5 — devono fallire anche valori "quasi plausibili" ma fuori dal
     range fisico stretto), con `n_politropico` fuori range (es. 0.5 e
     4.0), e con `massa_nucleo_msun` fuori dal range Chandrasekhar (es.
     0.8 e 3.0); verificare che ciascuno sollevi `ValueError`.
5.6. Test `get_progenitor_by_id`: ritorna la stella corretta per id
     valido, solleva errore chiaro per id inesistente.
5.7. Test di tracciabilità del placeholder: verificare che il campo
     `fonte` di ciascun progenitore del catalogo di riferimento contenga
     la sottostringa "placeholder" (case-insensitive), cosicché questi
     dati non possano in futuro essere scambiati per dati reali del
     catalogo completo.
5.8. Test parsing CSV con commento in testa: verificare che
     `load_catalog_from_csv` legga correttamente il file anche con le
     righe di commento bibliografico all'inizio (3.2), senza errori di
     parsing e senza includere le righe di commento come dati.

### 6. Chiusura del ciclo
6.1. Eseguire `pytest tests/test_catalog.py`, riportare solo l'esito
     sintetico (n. test passati/falliti), non l'output verboso.
6.2. Eseguire `scripts/step1_demo.py` e verificare che l'output sia
     leggibile e corretto.
6.3. Passare la mano al subagent `critic-fisico` per un controllo mirato:
     plausibilità dei valori placeholder (Ye, n, densità centrale, massa
     nucleo vs soglia Chandrasekhar) rispetto ai range vincolanti fissati
     in questo piano e alla tabella vincoli di CLAUDE.md — dichiarare per
     ciascun punto verificato/violato/non verificabile automaticamente.
6.4. Passare la mano al subagent `reporter` per il resoconto sintetico
     del ciclo.
6.5. Aggiornare questo file: marcare Step 1 come completato in cima,
     aggiungere riga in "Log cicli" con data, esito, e nota esplicita che
     il catalogo è un placeholder in attesa dei file sorgente completi.
6.6. **STOP esplicito**: fermarsi qui e attendere conferma esplicita
     dell'utente prima di iniziare Step 2 (regola CLAUDE.md — ogni ciclo
     giornaliero si ferma dopo l'aggiornamento di STATUS.md e il report,
     non si procede automaticamente allo step successivo).

### Nota per gli step futuri (non bloccante per Step 1)
Da Step 2 in poi servirà un modulo `collasso/constants.py` dedicato alle
costanti fisiche e alle conversioni di unità (sistema cgs, Msun, km, ecc.),
per evitare che valori come `G`, `c`, `Msun_g`, fattori di conversione
km<->cm vengano duplicati o hardcoded in più moduli. Annotazione del
planner per pianificazione futura, non è richiesto crearlo in Step 1.

## Piano ciclo corrente (Step 2) — REV. 2 (corretto dopo review)

Decisioni vincolanti già prese con l'utente (non rimetterle in discussione,
sono vincolanti): equazione di Lane-Emden `(1/xi^2) d/dxi(xi^2 dtheta/dxi) =
-theta^n`, IC `theta(0)=1, theta'(0)=0`; singolarità in xi=0 aggirata con
innesco in serie a piccolo xi0, poi integrazione con
`scipy.integrate.solve_ivp` (RK45, tolleranze strette) come sistema del
primo ordine in `(theta, phi=theta')`; superficie xi1 (primo zero di theta)
trovata con un `event` terminale di `solve_ivp`. Validazione contro
soluzioni analitiche note: n=0 (`theta=1-xi^2/6`, `xi1=sqrt(6)`), n=1
(`theta=sin(xi)/xi`, `xi1=pi`), n=5 (`theta=(1+xi^2/3)^-1/2`, xi1 infinito —
confronto solo su theta(xi) fino a xi_max finito, non su xi1). Struttura
file concordata: `collasso/constants.py`, `collasso/lane_emden.py`,
`scripts/step2_demo.py`, `tests/test_lane_emden.py`. Nessuna nuova
dipendenza esterna: numpy/scipy/pytest già installati in Step 1.

**Nota di revisione**: il reviewer ha esaminato la REV. 1 di questo piano
(formule e tolleranze) e ha dato verdetto positivo, nessuna riscrittura
necessaria; ha richiesto 4 rifiniture puntuali, integrate qui sotto ai
punti 1.1, 1.5 (nuovo), 3.1, 3.5, 3.6 (nuovo) e 3.7 (già presente,
rinumerato, con nota aggiuntiva). Nessun'altra parte del piano è stata
modificata rispetto alla REV. 1.

### Vincoli di modello per questo step (da CLAUDE.md, non negoziabili)
- **Gravità**: SOLO newtoniana in questo step. Nessuna correzione
  relativistica sul nucleo va introdotta nel solver di Lane-Emden o in
  `physical_profile` — quella è responsabilità esclusiva dello Step 5.
- **EOS**: `n` (indice politropico) resta un valore FISSO per ogni singola
  soluzione risolta dal solver. Nessuna transizione a indice variabile
  non-relativistico -> ultra-relativistico in questo step — quella è
  responsabilità esclusiva dello Step 3 (EOS di Chandrasekhar).
- Nessun termine di raffreddamento neutrini, nessuna rotazione/campo
  magnetico introdotti (limiti già dichiarati ed esclusi a monte, invariati
  da Step 1).
- Il mismatch noto tra i 4 parametri del catalogo placeholder (rilevato dal
  critic-fisico in Step 1, vedi docstring `collasso/catalog.py`: la massa
  implicita da `(rho_c, R, n)` è ~5-6x superiore a `massa_nucleo_msun`) va
  **documentato esplicitamente nell'output** dello script demo, non
  nascosto né "corretto" scegliendo ad hoc un parametro diverso da quelli
  concordati. La coppia primaria per fissare la scala fisica è sempre
  `(densita_centrale_gcm3, massa_nucleo_msun)`; `raggio_iniziale_km` resta
  un cross-check indipendente, mai un vincolo simultaneo.

### 0. Costanti fisiche — `collasso/constants.py`
0.1. Creare il modulo con tre costanti cgs, valori fissati dal planner (non
     a discrezione del coder):
     - `G_CGS = 6.6743e-8` (cm^3 g^-1 s^-2, CODATA 2018). NOTA: non
       utilizzata dalla formula di `physical_profile` in questo step (che
       non richiede la costante di gravitazione, essendo basata solo sulla
       relazione massa-densità-geometria di Lane-Emden); inclusa ora per
       completezza della struttura concordata e riservata all'uso negli
       step futuri (es. Step 3, determinazione della costante politropica
       K). Il coder non deve inventarsi un uso artificiale di `G_CGS` in
       questo step solo per "giustificarne" la presenza.
     - `M_SUN_G = 1.98892e33` (g; da parametro solare nominale IAU 2015,
       `GM_sun/G`).
     - `KM_CM = 1.0e5` (cm per km).
0.2. Nessuna dipendenza da altri moduli del progetto; costanti come
     variabili modulo-livello con commento che ne cita la fonte.

### 1. Solver Lane-Emden — `collasso/lane_emden.py`
1.1. dataclass `LaneEmdenSolution` con campi: `n`, `xi` (`np.ndarray`),
     `theta` (`np.ndarray`), `dtheta_dxi` (`np.ndarray`), `xi1`
     (`float | None`), `dtheta_dxi_at_xi1` (`float | None`). I campi
     `xi1`/`dtheta_dxi_at_xi1` sono `None` quando non esiste superficie
     finita entro `xi_max` (caso n>=5): comportamento da documentare nel
     docstring, NON deve sollevare eccezione a questo livello.
     **Nota di documentazione obbligatoria (da review, punto 4a)**: il
     campo `theta` NON ha il clipping a `>=0` applicato — il clip
     `max(theta, 0.0)` esiste SOLO dentro il RHS dell'ODE (punto 1.3) e
     nel calcolo di `rho_gcm3` in `physical_profile` (punto 1.10), MAI
     sull'array `theta` restituito nella dataclass. Di conseguenza
     l'ultimo punto della griglia (xi=xi1) può contenere un residuo
     numerico leggermente negativo (~1e-10). Il coder deve scrivere
     questo esplicitamente nel docstring del campo `theta` della
     dataclass (non lasciarlo implicito), per evitare che codice futuro
     assuma `theta>=0` ovunque — vedi anche verifica di test in 3.5.
1.2. Innesco in `xi0` fissato = `1e-4` (valore fissato dal planner):
     - `theta(xi0) = 1 - xi0**2/6 + n*xi0**4/120`
     - `dtheta/dxi(xi0) = -xi0/3 + n*xi0**3/30`
     (sviluppo in serie troncato al secondo ordine, sufficiente per
     `xi0=1e-4`; formule da usare esattamente come scritte, non da
     ri-derivare).
1.3. Sistema del primo ordine `(theta, phi=dtheta/dxi)`:
     - `dtheta/dxi = phi`
     - `dphi/dxi = -(2/xi)*phi - max(theta, 0.0)**n`
     Il clipping `max(theta, 0.0)` è **obbligatorio** e uniforme per ogni
     n (anche n=0,1, per coerenza di implementazione): evita basi negative
     elevate a esponente non intero (NaN/complesso) quando `theta` scende
     leggermente sotto zero per errore numerico vicino a `xi1`.
1.4. Integrazione con `scipy.integrate.solve_ivp`, parametri fissati dal
     planner (non lasciati a discrezione del coder):
     - `method="RK45"`
     - `rtol=1e-10`, `atol=1e-12` (tolleranze del SOLVER — distinte dalle
       tolleranze di TEST del punto 3, non confonderle)
     - `dense_output=True`
     - evento `theta == 0` (`direction=-1`, `terminal=True`) per
       determinare `xi1` quando esiste
     - `xi_max` esposto come parametro con default `50.0`; per n=5 va
       usato esplicitamente `xi_max=50.0` (coerente con le soluzioni
       analitiche concordate)
1.5. **(NUOVO, da review, punto 1)** Subito dopo la chiamata a
     `solve_ivp`, verificare esplicitamente `sol.success` (equivalente a
     `sol.status != -1`), PRIMA di processare l'evento o costruire
     `LaneEmdenSolution`:
     - se `sol.success` è `False`: sollevare `RuntimeError` con messaggio
       esplicito che includa `sol.message` (es. `f"integrazione Lane-Emden
       fallita per n={n}: {sol.message}"`), senza costruire alcun oggetto
       risultato parziale;
     - motivazione (da riportare nel commento del codice): senza questo
       controllo, un fallimento silenzioso di RK45 vicino alla rigidità
       numerica a xi~xi1 produrrebbe una soluzione apparentemente valida
       ma costruita su un'integrazione parziale/incompleta — il check
       deve essere incondizionato, non solo in modalità debug.
1.6. (era 1.5) Se l'evento scatta: `xi1` = ascissa dell'evento,
     `dtheta_dxi_at_xi1` = `phi` all'evento. Se NON scatta entro `xi_max`:
     `xi1=None`, `dtheta_dxi_at_xi1=None`.
1.7. (era 1.6) Griglia di output: `xi = np.linspace(xi0, xi1 if xi1 is not
     None else xi_max, n_points)` con `n_points` esposto come parametro,
     default fissato = `2000`; `theta`, `dtheta_dxi` valutati tramite
     `sol.sol(xi)` (dense output della stessa integrazione), MAI tramite
     una nuova integrazione separata.
1.8. (era 1.7) Validare `n >= 0` in ingresso (n negativo non fisico):
     `ValueError` esplicito se violato.
1.9. (era 1.8) `theta_analytic(xi, n)` e `xi1_analytic(n)`: implementate
     SOLO per n=0 (`theta=1-xi**2/6`, `xi1=sqrt(6)`) e n=1
     (`theta=sin(xi)/xi`, `xi1=pi`). Per qualunque altro n sollevare
     `NotImplementedError` con messaggio esplicito che rimanda al
     confronto per n=5, gestito separatamente e SOLO inline in demo/test
     (vedi 2.3 e 3.4) — non aggiungere qui un ramo "nascosto" per n=5,
     per non creare l'illusione di un `xi1` finito che non esiste.
1.10. (era 1.9) `physical_profile(solution, rho_c_gcm3, massa_msun) ->
     PhysicalProfile` (dataclass nello stesso modulo, campi: `n`,
     `rho_c_gcm3`, `massa_msun`, `alpha_cm`, `R_cm`, `R_km`, `r_cm`
     (`ndarray`), `rho_gcm3` (`ndarray`)):
     - se `solution.xi1 is None`: `ValueError` esplicito ("profilo fisico
       non definibile: nessuna superficie finita per n={n} entro xi_max,
       atteso per n>=5");
     - risolvere `alpha_cm` da
       `massa_msun*M_SUN_G = 4*pi*alpha_cm**3*rho_c_gcm3*(-xi1**2*dtheta_dxi_at_xi1)`
       cioè
       `alpha_cm = (massa_msun*M_SUN_G / (4*pi*rho_c_gcm3*(-xi1**2*dtheta_dxi_at_xi1)))**(1/3)`;
     - guardia: se `(-xi1**2*dtheta_dxi_at_xi1) <= 0` → `ValueError`
       (condizione non fisica, controllo difensivo);
     - `R_cm = alpha_cm*xi1`; `R_km = R_cm/KM_CM`;
     - `r_cm = alpha_cm*solution.xi`;
       `rho_gcm3 = rho_c_gcm3*np.maximum(solution.theta, 0.0)**n` (stesso
       clipping del punto 1.3, per coerenza);
     - `massa_msun` passato a questa funzione deve SEMPRE essere
       `massa_nucleo_msun` del catalogo quando chiamata dallo script demo
       (mai `massa_zams_msun` — vedi 2.5).

### 2. Script demo — `scripts/step2_demo.py`
2.1. Validazione n=0: `solve_lane_emden(0)`; confronto `theta` numerico vs
     `theta_analytic` su tutta la griglia `xi` restituita con
     `np.allclose(rtol=1e-6, atol=1e-8)` (stessa tolleranza di TEST fissata
     in 3.1, per coerenza demo/test); stampare una riga con scarto massimo
     assoluto e relativo, `xi1` numerico vs `sqrt(6)`, scarto.
2.2. Validazione n=1: stesso schema, `theta_analytic=sin(xi)/xi`, `xi1`
     numerico vs `pi`.
2.3. Validazione n=5: `solve_lane_emden(5, xi_max=50.0)`; confronto `theta`
     numerico vs `(1+xi**2/3)**-0.5` calcolato **inline** (NON tramite
     `theta_analytic`, che non supporta n=5 per decisione 1.9) su tutta la
     griglia restituita; stampare scarto massimo assoluto/relativo;
     stampare esplicitamente la riga "xi1 non definito per n=5 (nessuna
     superficie finita entro xi_max=50, atteso)".
2.4. Nessun output verboso: solo le righe di riepilogo dei 3 confronti
     sopra (regola CLAUDE.md sul filtraggio dell'output numerico).
2.5. Profilo fisico della stella scelta: caricare il catalogo di
     riferimento, `get_progenitor_by_id(catalog, "s20")`;
     `solve_lane_emden(n=stella.n_politropico)` (n=2.0 per s20);
     `physical_profile(solution, stella.densita_centrale_gcm3,
     stella.massa_nucleo_msun)` — **usare `massa_nucleo_msun`, mai
     `massa_zams_msun`**.
2.6. Stampare: `R_derivato_km`, `raggio_iniziale_km` (dal catalogo), scarto
     percentuale = `100*|R_derivato_km - raggio_iniziale_km|/raggio_iniziale_km`.
2.7. Stampare **esplicitamente** una riga di disclaimer che lo scarto è
     atteso/noto per il catalogo placeholder (richiamando la nota del
     critic-fisico di Step 1 in `collasso/catalog.py`) e NON un difetto del
     solver di Lane-Emden — riga obbligatoria, non opzionale.

### 3. Test — `tests/test_lane_emden.py`
3.1. Costanti di tolleranza da definire in testa al file (valori fissati
     dal planner, NON a discrezione del coder):
     - `TOL_THETA_RTOL = 1e-6`, `TOL_THETA_ATOL = 1e-8` → usate per n=0 e
       n=1 via `np.allclose(theta_num, theta_an, rtol=TOL_THETA_RTOL,
       atol=TOL_THETA_ATOL)`.
     - `TOL_XI1_ABS = 1e-5` → `abs(xi1_num - xi1_analitico) < TOL_XI1_ABS`,
       per n=0 e n=1.
     - `TOL_THETA_N5_RTOL = 1e-5`, `TOL_THETA_N5_ATOL = 1e-7` → confronto
       n=5 su `xi` in `[xi0, 50]`; tolleranza più larga perché il dominio
       di integrazione è più esteso.
     - `TOL_THETA0_ABS = 1e-6` → `abs(solution.theta[0] - 1.0) <
       TOL_THETA0_ABS` (riferito al primo punto di griglia `xi0=1e-4`, non
       a `xi=0` esatto — la soluzione non include mai `xi=0` per costruzione).
     - `TOL_MONOTONE = 1e-10` → `np.all(np.diff(solution.theta) <=
       TOL_MONOTONE)` (piccola slack positiva per rumore numerico su una
       sequenza attesa non-crescente).
     - `XI1_SANITY_MAX = 10.0` → limite superiore largo e sicuro per `xi1`
       con n in `[1.5, 3.0]` (letteratura: `xi1` tipico tra ~2 e ~7 in
       questo range). Controllo di plausibilità grossolano — non sostituisce
       il confronto quantitativo del punto 3.6 (nuovo, da review).
     - `TOL_THETA_MIN_ABS = -1e-8` → **(NUOVO, da review, punto 4a)** soglia
       inferiore ammessa per `solution.theta.min()`: `theta` pubblico NON
       ha il clipping applicato (vedi nota in 1.1), quindi può presentare
       un residuo numerico leggermente negativo (~1e-10) vicino a `xi1`;
       il test verifica `theta.min() > TOL_THETA_MIN_ABS`, non
       `theta.min() >= 0`.
     - `XI1_LITERATURE_RTOL = 1e-3` → **(NUOVO, da review, punto 3)**
       tolleranza relativa larga per il confronto quantitativo di `xi1`
       numerico contro i valori tabulati di letteratura (Chandrasekhar
       1939 / Horedt 2004) per n in {1.5, 2.0, 2.5, 3.0} — vedi test 3.6
       (nuovo).
     - `TOL_ALPHA_RTOL = 1e-6` → per il test di autoconsistenza di
       `physical_profile`.
3.2. `test_n0_matches_analytic`: confronto `theta` (regola 3.1), confronto
     `xi1` vs `sqrt(6)` (`TOL_XI1_ABS`).
3.3. `test_n1_matches_analytic`: confronto `theta` vs `sin(xi)/xi`,
     confronto `xi1` vs `pi`.
3.4. `test_n5_no_finite_surface_matches_analytic`: `solve_lane_emden(5,
     xi_max=50.0)`; `assert solution.xi1 is None` e
     `assert solution.dtheta_dxi_at_xi1 is None`; confronto `theta` vs
     `(1+xi**2/3)**-0.5` (calcolato inline nel test) con
     `TOL_THETA_N5_RTOL`/`TOL_THETA_N5_ATOL`.
3.5. `test_solver_properties` (parametrizzato su `n` in `[1.5, 2.0, 2.5,
     3.0]` — **esteso a 3.0 da review, punto 2**: 3.0 è il limite
     superiore del range fisico vincolante del catalogo (`n_politropico`
     1.5–3.0, fissato in Step 1), caso ultra-relativistico, prima non
     coperto dal test):
     - `theta[0]` vicino a 1 (`TOL_THETA0_ABS`);
     - `theta` non-crescente (`TOL_MONOTONE`);
     - `xi1 is not None`, `xi1 > 0`, `xi1 < XI1_SANITY_MAX`;
     - **(NUOVO, da review, punto 4a)** `theta.min() > TOL_THETA_MIN_ABS`
       (NON `theta.min() >= 0`): l'array `theta` pubblico non ha il
       clipping applicato (vedi nota in 1.1), quindi può presentare un
       residuo numerico leggermente negativo (~1e-10) all'ultimo punto di
       griglia (xi=xi1); il test deve includere un commento nel codice
       che spiega esplicitamente questo motivo, per non far sembrare la
       tolleranza un errore di battitura.
3.6. **(NUOVO, da review, punto 3)** `test_xi1_literature_values`:
     validazione quantitativa di `xi1` numerico contro valori tabulati di
     letteratura, complementare (non sostitutiva) al controllo di
     plausibilità grossolano `XI1_SANITY_MAX` del punto 3.5. Parametrizzato
     su n in {1.5, 2.0, 2.5, 3.0} — fonte bibliografica da citare
     esplicitamente nel commento del test (Chandrasekhar, *An Introduction
     to the Study of Stellar Structure*, 1939; valori riportati anche in
     Horedt, *Polytropes: Applications in Astrophysics and Related
     Fields*, 2004):
     - n=1.5 -> xi1 ≈ 3.65375
     - n=2.0 -> xi1 ≈ 4.35287
     - n=2.5 -> xi1 ≈ 5.35528
     - n=3.0 -> xi1 ≈ 6.89685
     `assert abs(xi1_num - xi1_letteratura) / xi1_letteratura <
     XI1_LITERATURE_RTOL`.
3.7. (era 3.6) `test_physical_profile_self_consistent`: `n_test=1.5`
     (fissato); `solution = solve_lane_emden(1.5)`; `alpha_test_cm=1.0e8`,
     `rho_c_test=1.0e10` (valori fissati dal planner, g/cm^3 e cm);
     `M_theoretical_g = 4*pi*alpha_test_cm**3*rho_c_test*
     (-solution.xi1**2*solution.dtheta_dxi_at_xi1)`;
     `M_theoretical_msun = M_theoretical_g / M_SUN_G`;
     `profile = physical_profile(solution, rho_c_test, M_theoretical_msun)`;
     `assert abs(profile.alpha_cm - alpha_test_cm)/alpha_test_cm <
     TOL_ALPHA_RTOL`.
     **Nota esplicita da aggiungere come commento nel test (da review,
     punto 4b)**: questo è un controllo di correttezza ALGEBRICA
     dell'inversione in `physical_profile` (radice cubica, segni,
     fattori) — il valore `M_theoretical` è calcolato con la STESSA
     formula usata poi in produzione dentro `physical_profile`, quindi il
     test NON costituisce una validazione fisica indipendente della
     formula stessa (non testa se la formula sia fisicamente corretta,
     solo se l'implementazione la inverte correttamente). Va dichiarato
     esplicitamente per evitare che in futuro venga scambiato per una
     verifica più forte di quanto sia.
3.8. (era 3.7) `test_physical_profile_raises_without_finite_surface`:
     `solve_lane_emden(5, xi_max=50.0)` (xi1 None) →
     `pytest.raises(ValueError)` chiamando `physical_profile`.
3.9. (era 3.8) `test_negative_n_raises`: `solve_lane_emden(-1)` →
     `pytest.raises(ValueError)`.

### 4. Chiusura del ciclo
4.1. Eseguire `pytest tests/test_lane_emden.py` (ed eventualmente l'intera
     suite `tests/`), riportare solo l'esito sintetico (n. test
     passati/falliti), non il log verboso.
4.2. Eseguire `scripts/step2_demo.py`, verificare che l'output sia
     leggibile, filtrato, e contenga i 3 confronti analitici (n=0,1,5) più
     il disclaimer obbligatorio sul mismatch del catalogo (2.7).
4.3. Passare la mano al subagent `critic-fisico` con checklist mirata:
     - gravità solo newtoniana (nessuna correzione relativistica introdotta
       impropriamente in questo step);
     - `n` fisso per ogni soluzione (nessuna anticipazione della EOS a
       indice variabile di Step 3);
     - nessun termine neutrini/rotazione introdotto;
     - condizioni al contorno di Lane-Emden `theta(0)=1`, `theta'(0)=0`
       rispettate entro le tolleranze fissate al punto 3.1;
     - il mismatch massa/raggio noto (rilevato in Step 1) è dichiarato
       esplicitamente nell'output dello script demo, non "aggiustato"
       scegliendo un parametro diverso da `(rho_c, massa_nucleo_msun)`
       come coppia primaria;
     - il check `sol.success`/`RuntimeError` (punto 1.5) è implementato
       incondizionatamente, non solo come commento o TODO.
4.4. Passare la mano al subagent `reporter` per il resoconto sintetico del
     ciclo.
4.5. Aggiornare questo file: marcare Step 2 come completato in cima,
     aggiungere riga in "Log cicli" con data, esito test, esito
     critic-fisico, e nota sugli scarti osservati (validazione analitica e
     scarto raggio derivato vs catalogo).
4.6. **STOP esplicito**: fermarsi qui e attendere conferma esplicita
     dell'utente prima di iniziare Step 3 (stessa regola CLAUDE.md già
     applicata a Step 1 — nessun avanzamento automatico allo step
     successivo).

## Piano ciclo corrente (Step 3) — REV. 2 (corretto dopo review)

Approccio già approvato dall'utente (NON rimetterlo in discussione, è
vincolante — formule e costanti fissate, non a discrezione del coder):
EOS esatta del gas di elettroni degenere (Chandrasekhar 1939; Shapiro &
Teukolsky 1983, cap. 2-3):
- `x = (hbar/(m_e*c)) * (3*pi^2*n_e)^(1/3)`, `n_e = Ye * rho / m_u`
- `f(x) = x*(2*x^2-3)*sqrt(1+x^2) + 3*asinh(x)`
- `P(x) = (m_e^4*c^5)/(24*pi^2*hbar^3) * f(x)`
- limite NR (x->0): `P_NR = ((3*pi^2)^(2/3)/5) * (hbar^2/m_e) * n_e^(5/3)`
  (∝ rho^(5/3), n=3/2)
- limite UR (x->∞): `P_UR = ((3*pi^2)^(1/3)/4) * hbar*c * n_e^(4/3)`
  (∝ rho^(4/3), n=3)
- `Gamma1(x) = (8*x^5)/(3*f(x)*sqrt(1+x^2))`, `n_eff(x) = 1/(Gamma1(x)-1)`
  (limiti: 5/3 / n=3/2 per x->0; 4/3 / n=3 per x->∞)
- `K_UR(Ye) = ((3*pi^2)^(1/3)/4) * hbar*c * (Ye/m_u)^(4/3)`, tale che
  `P_UR = K_UR(Ye)*rho^(4/3)`
- `M_Ch = 4*pi*(K_UR(Ye)/(pi*G))^(3/2) * (-xi1^2*theta'(xi1))`, con `xi1` e
  `theta'(xi1)` presi dalla soluzione REALE di
  `collasso.lane_emden.solve_lane_emden(3.0)` (riuso diretto del solver di
  Step 2, mai una costante tabulata a mano).

Costanti nuove fissate dal planner (CODATA 2018) da aggiungere a
`collasso/constants.py`: `HBAR_CGS = 1.054571817e-27` (erg*s), `C_LIGHT_CGS
= 2.99792458e10` (cm/s), `M_ELECTRON_G = 9.1093837015e-28` (g), `M_U_G =
1.66053906660e-24` (g, unità di massa atomica unificata, NON `m_H`).

File concordati: `collasso/eos.py`, `scripts/step3_demo.py`,
`tests/test_eos.py`.

**Nota di revisione**: il reviewer ha verificato analiticamente tutte le
formule fisiche e le costanti CODATA di questo piano (tutte corrette,
nessun errore) ma ha segnalato un problema numerico serio e concreto: la
valutazione diretta di `chandrasekhar_f(x) = x*(2*x^2-3)*sqrt(1+x^2) +
3*asinh(x)` per `x` piccolo (es. `x=1e-4`, il punto di test degli
asintoti già fissato in 3.1/3.2) soffre di **cancellazione catastrofica**
in floating point: i due termini della somma valgono ciascuno ~∓3e-4
mentre il risultato atteso è ~1.6e-20 (ordine `x^5`), con un numero di
condizionamento che supera `1/eps` della precisione double — l'errore
relativo atteso sul valore calcolato con la formula diretta è enorme
(anche >100%), non l'`1e-6` fissato come tolleranza di test. Questo
comprometterebbe il test sugli asintoti (3.2), il test su `Gamma1(x)`
(3.3, che chiama `chandrasekhar_f`), il cross-check `P_esatta` vs
`P_NR`/`P_UR` (3.9), e la prima riga della tabella dello script demo
(2.1-2.2, che parte da `x=1e-4`). Il piano è stato corretto qui sotto
imponendo un'implementazione **a due rami**, numericamente stabile, per
`chandrasekhar_f` (punto 1.2 aggiornato), con propagazione automatica
della correzione a `_gamma1_of_x`/`gamma1_chandrasekhar` (punto 1.7,
nota aggiunta) e un nuovo test dedicato di continuità del branching
(punto 3.10, nuovo). Nessun'altra parte del piano è stata modificata:
formule fisiche, costanti CODATA, tolleranze di `M_Ch`
(`TOL_MCH_MIN_MSUN`/`TOL_MCH_MAX_MSUN`) e del cross-check a differenze
finite di `Gamma1` (`H_FD_REL`/`TOL_GAMMA1_FD_RTOL`), e perimetro dello
step restano quelli già validati dal reviewer.

### Vincoli di modello per questo step (da CLAUDE.md, non negoziabili)
- EOS di gas completamente degenere e **freddo (T=0)**: nessun termine
  termico/entropico, nessuna dipendenza dalla temperatura.
- Questo step fornisce **SOLO** la funzione `P(rho)` (e `Gamma1(rho)`,
  `n_eff(rho)`, `M_Ch(Ye)`): NON reimposta l'equilibrio stellare (resta
  responsabilità di Lane-Emden/Step 2), NON introduce dinamica (Step 4),
  NON introduce correzioni relativistiche sul nucleo oltre alla EOS stessa
  (Step 5), NON introduce il limite TOV/classificazione remnant (Step 6).
- `chandrasekhar_mass_msun(ye)` è la massa di Chandrasekhar **Newtoniana
  classica** per un politropo n=3 (indipendente da `rho_c`) — va
  dichiarata esplicitamente come tale nel codice e nell'output demo, mai
  presentata come "la" soglia definitiva di instabilità GR-corretta
  (quella resta esclusivamente il limite di Oppenheimer-Volkoff, Step 6).

### 0. Costanti fisiche aggiuntive — `collasso/constants.py`
0.1. Aggiungere, come variabili modulo-livello con commento di fonte
     (CODATA 2018), senza toccare le costanti già presenti (`G_CGS`,
     `M_SUN_G`, `KM_CM`, invariate):
     - `HBAR_CGS = 1.054571817e-27`
     - `C_LIGHT_CGS = 2.99792458e10`
     - `M_ELECTRON_G = 9.1093837015e-28`
     - `M_U_G = 1.66053906660e-24`
0.2. `G_CGS` viene finalmente utilizzata in questo step da
     `chandrasekhar_mass_msun` — chiude esplicitamente la nota lasciata
     nel commento di Step 2 ("riservata all'uso negli step futuri, es.
     Step 3"); aggiungere una riga di commento in `constants.py` che lo
     conferma (non è obbligatorio, ma consigliato per tracciabilità).
0.3. Nessuna dipendenza tra le nuove costanti e altri moduli del progetto
     (stesso principio di modularità di Step 2).

### 1. Modulo EOS — `collasso/eos.py`
1.1. `fermi_x(rho_gcm3, ye)`: implementa `x = (hbar/(m_e*c)) *
     (3*pi^2*n_e)^(1/3)` con `n_e = ye*rho_gcm3/M_U_G`, usando le costanti
     di `collasso.constants`. Deve accettare sia scalari `float` sia
     `np.ndarray` (vettorizzata, nessun loop esplicito — necessario per le
     tabelle a griglia dello script demo).
1.2. `chandrasekhar_f(x)`: building-block **puro su x** (nessuna
     dipendenza da `rho`/`ye`). **REV. 2 — implementazione a due rami,
     fissata dal planner dopo review numerica (non a discrezione del
     coder), per correggere una cancellazione catastrofica identificata
     dal reviewer**:
     - Costante di modulo `X_SERIES_THRESHOLD = 0.1`, definita in testa a
       `collasso/eos.py` (accanto a `_P0_CGS`).
     - Per `x < X_SERIES_THRESHOLD`: usare lo sviluppo in serie troncato
       (Taylor di `f(x)` attorno a `x=0`; derivato e verificato termine
       per termine dal planner contro la forma diretta e contro il
       riferimento di letteratura
       `f(x)=(8/5)x^5*[1-(5/14)x^2+(5/24)x^4-...]`):
       `f_series(x) = (8/5)*x**5 - (4/7)*x**7 + (1/3)*x**9`
       (troncamento a 3 termini; a `x=X_SERIES_THRESHOLD` l'errore di
       troncamento è dell'ordine di `1e-11` relativo, trascurabile
       rispetto a qualunque tolleranza di test fissata in questo piano —
       nessun termine ulteriore necessario). Formula da usare esattamente
       come scritta, non da ri-derivare.
     - Per `x >= X_SERIES_THRESHOLD`: formula diretta esatta, invariata
       rispetto alla REV. 1: `f(x) = x*(2*x^2-3)*sqrt(1+x^2) +
       3*np.arcsinh(x)` (a `x=X_SERIES_THRESHOLD` la formula diretta è
       già accurata entro ~`1e-11` relativo rispetto alla serie, quindi
       non c'è discontinuità significativa al bordo — nessun altro fix
       necessario lì; verificato comunque da un test dedicato, 3.10).
     - **Motivazione del branching, da riportare nel docstring della
       funzione (non solo in questo piano)**: per `x` piccolo la formula
       diretta somma due termini di segno opposto e modulo quasi uguale
       (`x*(2*x^2-3)*sqrt(1+x^2) ~ -3x`, `3*asinh(x) ~ +3x`), il cui
       risultato atteso è `O(x^5)` — la sottrazione fra numeri quasi
       uguali cancella molte cifre significative e l'errore relativo sul
       valore calcolato con la formula diretta esplode ben oltre la
       precisione macchina utile (**cancellazione catastrofica**, non un
       problema di overflow/underflow né di `asinh` in sé). Il ramo a
       serie evita del tutto la sottrazione fra termini quasi opposti.
     - Guardia: `ValueError` se `x < 0` in qualunque elemento (analoga
       alla guardia `n>=0` di Lane-Emden Step 2), applicata PRIMA del
       branching serie/diretta.
     - Vettorizzata (numpy): il branching deve essere elementwise, es.
       `np.where(x < X_SERIES_THRESHOLD, f_series(x), f_direct(x))`
       (calcolare entrambi i rami sull'intero array e selezionare con
       `np.where` è accettabile anche se leggermente ridondante dal punto
       di vista computazionale — preferibile a un loop Python esplicito,
       per coerenza con il resto del modulo, vedi 1.11).
1.3. Prefattore `_P0_CGS` (costante privata di modulo, calcolata a tempo
     di import dalle costanti fondamentali): `_P0_CGS =
     (M_ELECTRON_G**4 * C_LIGHT_CGS**5) / (24.0 * math.pi**2 *
     HBAR_CGS**3)` (erg/cm^3 per unità di `f(x)`).
1.4. `pressure_chandrasekhar(rho_gcm3, ye)`: `_P0_CGS *
     chandrasekhar_f(fermi_x(rho_gcm3, ye))`. È la EOS esatta, quella da
     usare come riferimento in tutti i confronti (mai `pressure_non_relativistic`
     o `pressure_ultrarelativistic` come "verità" oltre i rispettivi limiti).
1.5. `pressure_non_relativistic(rho_gcm3, ye)`: formula diretta in `n_e`
     (NON passa da `fermi_x`/`chandrasekhar_f`): `((3*pi^2)^(2/3)/5) *
     (HBAR_CGS**2/M_ELECTRON_G) * n_e**(5/3)`, `n_e = ye*rho_gcm3/M_U_G`.
1.6. `k_ultrarelativistic(ye)`: `((3*pi^2)^(1/3)/4) * HBAR_CGS *
     C_LIGHT_CGS * (ye/M_U_G)**(4/3)`.
     `pressure_ultrarelativistic(rho_gcm3, ye)`: `k_ultrarelativistic(ye) *
     rho_gcm3**(4/3)` (deve riusare `k_ultrarelativistic`, non duplicare
     la formula).
1.7. `gamma1_chandrasekhar(rho_gcm3, ye)`: calcola `x =
     fermi_x(rho_gcm3, ye)` e ritorna `_gamma1_of_x(x)`, dove
     `_gamma1_of_x(x)` è una **funzione privata pure-x** (nome fissato):
     `Gamma1(x) = (8*x^5)/(3*chandrasekhar_f(x)*sqrt(1+x^2))`. Questo
     helper privato NON è nella lista pubblica concordata col coder, ma è
     necessario per poter testare gli asintoti "in forma pura su x"
     richiesti esplicitamente (vedi 3.3) — convenzione di naming coerente
     con i privati già presenti in `lane_emden.py` (`_lane_emden_rhs`,
     `_theta_zero_event`).
     **Nota REV. 2**: `_gamma1_of_x` chiama `chandrasekhar_f(x)` così
     com'è, branching incluso (punto 1.2 aggiornato) — non serve un ramo
     separato per `Gamma1`: la stabilità numerica del ramo a serie di
     `chandrasekhar_f` per `x` piccolo si propaga correttamente al
     rapporto (a numeratore `8*x^5` e denominatore `3*f(x)*sqrt(1+x^2)`
     entrambi ben condizionati quando `f(x)` è calcolata con la serie),
     nessuna nuova cancellazione viene introdotta nel quoziente stesso.
1.8. `n_eff_chandrasekhar(rho_gcm3, ye)`: `1.0 /
     (gamma1_chandrasekhar(rho_gcm3, ye) - 1.0)`.
1.9. `chandrasekhar_mass_msun(ye)`: chiama
     `collasso.lane_emden.solve_lane_emden(3.0)` (riuso diretto, import
     esplicito dal modulo Step 2, mai una costante `xi1`/`theta'(xi1)`
     ricopiata a mano — il valore atteso `xi1≈6.89685` è lo stesso già
     validato contro letteratura nel test Step 2, `test_xi1_literature_values`);
     guardia difensiva: se `solution.xi1 is None` → `RuntimeError`
     esplicito (non dovrebbe mai accadere per n=3 con `xi_max` default
     50.0, ma il check è comunque obbligatorio, stesso principio del check
     `sol.success` di Step 2); calcola `M_Ch_g = 4*pi*(k_ultrarelativistic(ye)
     /(pi*G_CGS))**1.5 * (-solution.xi1**2*solution.dtheta_dxi_at_xi1)`;
     ritorna `M_Ch_g / M_SUN_G`.
1.10. Validazione input, uniforme su tutte le funzioni che la richiedono:
      - `rho_gcm3 > 0` **strettamente** (scalare o elementwise su
        `np.ndarray`) → altrimenti `ValueError` esplicito con range atteso;
      - `0 < ye <= 1` (range fisico **generale** dell'EOS, NON il range
        catalogo-specifico 0.42–0.50 di `collasso.catalog` — l'EOS è una
        funzione generale del gas di elettroni degenere, non limitata alle
        stelle del catalogo di riferimento) → altrimenti `ValueError`
        esplicito.
      - Nota di design da riportare nel commento del codice: richiedere
        `rho_gcm3 > 0` stretto evita naturalmente la forma indeterminata
        `0/0` di `Gamma1` in `x=0` (`f(0)=0`, `Gamma1` non definita lì),
        senza bisogno di una guardia dedicata separata.
1.11. Tutte le funzioni su `rho`/`x` devono essere vettorizzate (numpy),
      non solo scalari — richiesto dalle tabelle a griglia dello script
      demo (2.x) e da alcuni test (3.x).
1.12. Formula di inversione `x -> rho` (fissata dal planner, forma chiusa,
      **non** root-finding, da riusare identica in demo e test):
      `rho(x, ye) = M_U_G/(ye*3*pi^2) * (x*M_ELECTRON_G*C_LIGHT_CGS/HBAR_CGS)**3`.
      Collocazione a discrezione del coder (funzione privata di supporto
      in `eos.py`, es. `_rho_from_x`, oppure calcolo inline in demo/test);
      la formula stessa NON è a discrezione del coder.

### 2. Script demo — `scripts/step3_demo.py`
2.1. Griglia `x = np.logspace(-4, 4, 17)` (un punto a decade, da 1e-4 a
     1e4, attraversa la transizione); `ye = 0.5` rappresentativo;
     `rho = _rho_from_x(x, 0.5)` (formula 1.12).
2.2. Tabella (17 righe) con colonne: `rho`, `x`, `P_esatta`
     (`pressure_chandrasekhar`), `P_NR`, scarto relativo `%` da `P_esatta`,
     `P_UR`, scarto relativo `%` da `P_esatta`, `n_eff`
     (`n_eff_chandrasekhar`) — stampata in formato compatto, non verboso.
     **Nota REV. 2**: la prima riga (`x=1e-4`) è ora affidabile grazie al
     ramo a serie di `chandrasekhar_f` (punto 1.2) — senza quel fix
     avrebbe mostrato un valore di `P_esatta` inquinato da cancellazione
     catastrofica.
2.3. Evidenziare esplicitamente (riga di commento in output) i punti dove
     `x ~ 1` (regione di transizione): qui `P_esatta` si discosta sia da
     `P_NR` sia da `P_UR` in modo significativo — atteso, da dichiarare
     come comportamento corretto, non un errore del calcolo.
2.4. Stampare esplicitamente la transizione di `n_eff` lungo la griglia
     (da ~1.5 a ~3), come richiesto dal task.
2.5. `chandrasekhar_mass_msun` per `ye=0.5` **e** per i tre `ye` del
     catalogo di riferimento, letti da
     `collasso.catalog.load_reference_catalog()` (mai ricopiati a mano):
     s15 (ye=0.50), s20 (ye=0.46), s25 (ye=0.43). Stampare per ciascuno:
     `id`, `ye`, `M_Ch(ye)` in Msun.
2.6. Confronto esplicito col valore "~1.4 Msun" di CLAUDE.md: scarto
     percentuale `100*|M_Ch(ye)-1.4|/1.4` per ciascun `ye`.
2.7. Riga di disclaimer **obbligatoria** (non opzionale, stesso principio
     del disclaimer di Step 2 sul mismatch catalogo): dichiarare
     esplicitamente che `chandrasekhar_mass_msun` è il valore Newtoniano
     classico per politropo n=3 (EOS completamente degenere, T=0), NON la
     soglia GR-corretta (limite di Oppenheimer-Volkoff/TOV, Step 6).
2.8. Nessun output verboso: solo le tabelle/righe sopra indicate (regola
     CLAUDE.md sul filtraggio dell'output numerico).

### 3. Test — `tests/test_eos.py`
3.1. Costanti di tolleranza fissate dal planner, in testa al file (NON a
     discrezione del coder):
     - `TOL_ASYMPT_RTOL = 1e-6` → usata per **tutti e 4** i controlli
       asintotici pure-x di 3.2/3.3 (`chandrasekhar_f` a `x=1e-4` e
       `x=1e4`; `_gamma1_of_x` a `x=1e-4` e `x=1e4`). Motivazione numerica
       (sviluppo in serie verificato dal planner): a `x=1e-4` lo scarto
       relativo atteso di `f(x)/x^5` da `8/5` e di `Gamma1(x)` da `5/3` è
       `O(x^2) ~ 1e-8`; a `x=1e4` lo scarto relativo atteso di `f(x)/x^4`
       da `2` e di `Gamma1(x)` da `4/3` è `O(1/x^2) ~ 1e-8`; `1e-6` lascia
       due ordini di grandezza di margine rispetto allo scarto atteso.
       **Nota REV. 2 (fix cancellazione catastrofica)**: questa tolleranza
       al punto di test `x=1e-4` è valida SOLO grazie al ramo a serie di
       `chandrasekhar_f` fissato al punto 1.2 — senza quel ramo la formula
       diretta soffrirebbe di cancellazione catastrofica a `x` piccolo,
       con errore relativo atteso anche >100%, ben oltre `1e-6`. Il punto
       di test `x=1e-4` resta invariato (non va spostato): con il ramo a
       serie è ora affidabile.
     - `TOL_BRANCH_CONTINUITY_RTOL = 1e-6` → **(NUOVO, REV. 2, fix
       cancellazione catastrofica)** tolleranza relativa per il test di
       continuità/correttezza del branching di `chandrasekhar_f` fra i due
       rami (serie vs diretta) appena sotto/sopra `X_SERIES_THRESHOLD` —
       vedi test 3.10 (nuovo).
     - `TOL_MCH_MIN_MSUN = 1.3`, `TOL_MCH_MAX_MSUN = 1.6` →
       `chandrasekhar_mass_msun(0.5)` deve cadere in `[1.3, 1.6]` Msun.
       Valore atteso dalla formula ≈1.458 Msun (calcolo di verifica del
       planner con le costanti CODATA fissate sopra), coerente con la nota
       relazione approssimata di letteratura `M_Ch ≈ 5.83*Ye^2` Msun
       (Shapiro & Teukolsky 1983, cap. 3), che per `Ye=0.5` dà ≈1.4575
       Msun — citare questa fonte nel commento del test.
     - `H_FD_REL = 1e-4`, `TOL_GAMMA1_FD_RTOL = 1e-5` → cross-check a
       differenze finite centrali in log-log dell'indice adiabatico:
       `Gamma1_fd = (ln P(rho*(1+H_FD_REL)) - ln P(rho*(1-H_FD_REL))) /
       ln((1+H_FD_REL)/(1-H_FD_REL))`, calcolato con
       `pressure_chandrasekhar` (EOS esatta, non i limiti NR/UR);
       valutato a `rho in {1e6, 1e7}` g/cm^3 con `ye=0.5`
       (corrispondenti a `x≈0.80` e `x≈1.72` per `ye=0.5` — punti scelti
       apposta in piena regione di transizione, non in un limite
       asintotico, per un test discriminante sulla derivata analitica di
       `Gamma1`); `assert abs(Gamma1_fd -
       gamma1_chandrasekhar(rho,0.5))/gamma1_chandrasekhar(rho,0.5) <
       TOL_GAMMA1_FD_RTOL`.
     - Monotonia di `pressure_chandrasekhar` in `rho`: griglia
       `np.logspace(2, 12, 50)` g/cm^3, `ye=0.5`; `assert
       np.all(np.diff(P) > 0)` (monotonia **stretta**, nessuna slack: a
       differenza del caso Lane-Emden non c'è integrazione ODE vicino a
       una singolarità, la funzione è analitica liscia in `rho` a `ye`
       fisso, quindi non ci si aspetta rumore numerico che violi la
       monotonia stretta su questa griglia).
3.2. `test_chandrasekhar_f_asymptotics`: `x=1e-4` → `assert
     abs(chandrasekhar_f(x)/x**5 - 8/5)/(8/5) < TOL_ASYMPT_RTOL`; `x=1e4`
     → `assert abs(chandrasekhar_f(x)/x**4 - 2)/2 < TOL_ASYMPT_RTOL`.
3.3. `test_gamma1_of_x_asymptotics` (import esplicito di `_gamma1_of_x`
     dal modulo, per testare "in forma pura su x" come richiesto): `x=1e-4`
     → `assert abs(_gamma1_of_x(x) - 5/3)/(5/3) < TOL_ASYMPT_RTOL`; `x=1e4`
     → `assert abs(_gamma1_of_x(x) - 4/3)/(4/3) < TOL_ASYMPT_RTOL`.
3.4. `test_pressure_chandrasekhar_monotonic_in_rho`: vedi formula/griglia
     in 3.1.
3.5. `test_chandrasekhar_mass_msun_ye_half`:
     `chandrasekhar_mass_msun(0.5)` in `[TOL_MCH_MIN_MSUN,
     TOL_MCH_MAX_MSUN]`.
3.6. `test_gamma1_finite_difference_crosscheck`: parametrizzato su `rho in
     {1e6, 1e7}` (ye=0.5), vedi formula/tolleranza in 3.1.
3.7. (aggiuntivo, buona pratica — stesso stile di validazione già usato in
     Step 1/2, non introduce nuove tolleranze fisiche, solo controlli
     binari su eccezioni) `test_invalid_rho_raises`: `rho<=0` (es. `0.0` e
     `-1.0`) → `pytest.raises(ValueError)` per `fermi_x`,
     `pressure_chandrasekhar`, `gamma1_chandrasekhar`.
3.8. (aggiuntivo, idem) `test_invalid_ye_raises`: `ye<=0` e `ye>1` (es.
     `0.0`, `-0.1`, `1.5`) → `pytest.raises(ValueError)` per `fermi_x`,
     `pressure_chandrasekhar`, `k_ultrarelativistic`,
     `chandrasekhar_mass_msun`.
3.9. (aggiuntivo, idem — cross-check di consistenza dei **prefattori**,
     non solo della forma di `chandrasekhar_f`) `test_pressure_matches_nr_and_ur_limits`:
     a `rho` tale che `x=1e-4` (`ye=0.5`, via `_rho_from_x`) → `assert
     abs(pressure_chandrasekhar(rho,0.5) -
     pressure_non_relativistic(rho,0.5)) / pressure_chandrasekhar(rho,0.5)
     < TOL_ASYMPT_RTOL`; a `rho` tale che `x=1e4` → scarto relativo
     analogo con `pressure_ultrarelativistic` `< TOL_ASYMPT_RTOL`. Verifica
     che i prefattori (non solo la forma adimensionale di
     `chandrasekhar_f`) siano numericamente coerenti tra le tre funzioni
     esposte. **Nota REV. 2**: questo test a `x=1e-4` è ora affidabile
     grazie al ramo a serie di `chandrasekhar_f` (punto 1.2) — prima della
     correzione, `pressure_chandrasekhar(rho,0.5)` a questo `x` sarebbe
     stata inquinata da cancellazione catastrofica.
3.10. **(NUOVO, REV. 2, fix cancellazione catastrofica)**
      `test_chandrasekhar_f_series_branch_continuity`: verifica dedicata
      del branching stesso di `chandrasekhar_f` attorno a
      `X_SERIES_THRESHOLD=0.1`, per intercettare eventuali errori di
      trascrizione della formula a serie o della soglia:
      - a `x=0.099` (appena SOTTO la soglia — la produzione usa il ramo a
        serie): calcolare nel test anche il valore da formula diretta
        `x*(2*x^2-3)*sqrt(1+x^2)+3*np.arcsinh(x)` (a `x~0.1` la formula
        diretta non soffre ancora di cancellazione catastrofica
        rilevante, errore atteso ~`1e-11` relativo, vedi nota 1.2) e
        confrontarlo con `chandrasekhar_f(0.099)` (ramo serie, prodotto
        dalla funzione reale): `assert abs(chandrasekhar_f(0.099) -
        f_diretta_test(0.099)) / f_diretta_test(0.099) <
        TOL_BRANCH_CONTINUITY_RTOL`;
      - a `x=0.101` (appena SOPRA la soglia — la produzione usa il ramo
        diretto): calcolare nel test anche il valore da formula a serie
        `(8/5)*x**5 - (4/7)*x**7 + (1/3)*x**9` e confrontarlo con
        `chandrasekhar_f(0.101)` (ramo diretto, prodotto dalla funzione
        reale): `assert abs(chandrasekhar_f(0.101) -
        f_serie_test(0.101)) / chandrasekhar_f(0.101) <
        TOL_BRANCH_CONTINUITY_RTOL`;
      - le formule di riferimento usate nel test (diretta e serie) vanno
        scritte esplicitamente nel test stesso (non importate come
        helper privati del modulo), in modo che il test sia un controllo
        indipendente della coerenza fra i due rami e non un semplice
        richiamo circolare alla stessa implementazione.

### 4. Chiusura del ciclo
4.1. Eseguire `pytest tests/` (intera suite, per verificare anche
     l'assenza di regressioni su Step 1/2), riportare solo l'esito
     sintetico (n. test passati/falliti), non il log verboso.
4.2. Eseguire `scripts/step3_demo.py`, verificare che l'output sia
     leggibile, filtrato, e contenga tutte le tabelle/righe di cui ai
     punti 2.1–2.7, incluso il disclaimer obbligatorio.
4.3. Passare la mano al subagent `critic-fisico` con checklist mirata:
     - EOS a T=0, nessun termine termico/entropico introdotto;
     - nessuna dinamica, nessuna correzione relativistica sul nucleo oltre
       alla EOS stessa, nessun limite TOV introdotti in questo step
       (violazione di scope verso Step 4/5/6);
     - formule esatte di Chandrasekhar (`f(x)`, `P(x)`, `Gamma1(x)`)
       verificate contro le fonti citate (Shapiro & Teukolsky 1983);
     - **(NUOVO, REV. 2)** implementazione a due rami di `chandrasekhar_f`
       (soglia `X_SERIES_THRESHOLD=0.1`, serie troncata a 3 termini per
       `x` piccolo) verificata: nessuna cancellazione catastrofica
       residua a `x` piccolo, branching documentato nel docstring con la
       motivazione (cancellazione, non overflow/underflow/asinh), test di
       continuità del branching (3.10) presente e passato;
     - `chandrasekhar_mass_msun` dichiarato esplicitamente come valore
       Newtoniano classico (non GR-corretto) sia nel codice sia
       nell'output demo;
     - riuso genuino di `solve_lane_emden(3.0)` (non una costante
       `xi1`/`theta'(xi1)` tabulata a mano) per `chandrasekhar_mass_msun`;
     - range di validazione `ye` dell'EOS (0,1] correttamente distinto dal
       range catalogo-specifico (0.42-0.50) usato in `collasso.catalog`,
       nessuna confusione tra i due nel codice.
4.4. Passare la mano al subagent `reporter` per il resoconto sintetico del
     ciclo.
4.5. Aggiornare questo file: marcare Step 3 come completato in cima,
     aggiungere riga in "Log cicli" con data, esito test, esito
     critic-fisico, e valori numerici chiave osservati (es.
     `chandrasekhar_mass_msun` per i 3 ye del catalogo, scarto dal valore
     "~1.4 Msun" di CLAUDE.md).
4.6. **STOP esplicito**: fermarsi qui e attendere conferma esplicita
     dell'utente prima di iniziare Step 4 (stessa regola CLAUDE.md già
     applicata a Step 1 e Step 2 — nessun avanzamento automatico allo step
     successivo).

## Piano ciclo corrente (Step 4) — REV. 3 (corretto dopo review — fix metrica test 3.5)

Approccio già approvato dall'utente (NON rimetterlo in discussione, è
vincolante — schema numerico, formule e struttura file fissati, riportati
qui integralmente per riferimento del coder):

Dinamica shell Lagrangiane a griglia sfalsata (stile von Neumann–Richtmyer),
gravità Newtoniana + gradiente di pressione dall'EOS esatta di
Chandrasekhar (Step 3, `pressure_chandrasekhar`). Nessun termine neutrini
(limite dichiarato, rimandato come le correzioni relativistiche a Step 5),
nessuna viscosità artificiale (limite dichiarato: possibile "ringing"
numerico in forte compressione, accettabile perché questo step arriva solo
all'innesco del collasso, non al bounce — **nota REV. 2**: il sintomo
estremo di questo limite, l'attraversamento fra shell consecutive, è ora
gestito con un evento terminale pulito di `solve_ivp` invece che con
un'eccezione sollevata dentro il RHS, vedi sotto).

N shell di massa fissa Δm_i (i=1..N, Lagrangiane: Δm_i costante nel tempo),
stato = raggi di bordo r_i(t) e velocità v_i(t)=dr_i/dt, r_0≡0 (centro,
sempre implicito, mai una variabile di stato).
- Massa racchiusa al bordo i (costante nel tempo): M_i = somma cumulativa
  di Δm_k per k=1..i
- Volume/densità shell i: V_i(t) = (4/3)π(r_i^3 - r_{i-1}^3),
  ρ_i(t) = Δm_i / V_i(t)
- Pressione: P_i(t) = pressure_chandrasekhar(ρ_i(t), Ye) (Ye fissato al
  valore catalogo della stella scelta)
- Moto bordo interno i=1..N-1: dv_i/dt = -G·M_i/r_i² -
  4π·r_i²·(P_{i+1}-P_i)/((Δm_i+Δm_{i+1})/2)
- Bordo esterno N (superficie libera, P_esterna=0): dv_N/dt = -G·M_N/r_N² -
  4π·r_N²·(0-P_N)/(Δm_N/2) = -G·M_N/r_N² + 4π·r_N²·P_N/(Δm_N/2)
- Integrazione: `scipy.integrate.solve_ivp`, sistema del primo ordine in
  (r_1..r_N, v_1..v_N), y0=(r_i(0), 0) (shell inizialmente a riposo)
- Evento terminale quando r_1(t) < r_min_frac·r_1(0) (soglia fissata sotto)

Condizione iniziale: da `collasso.lane_emden.physical_profile()` (Step 2,
r_cm/ρ_gcm3) si integra la massa cumulativa M(r) (trapezoidale), si
scelgono N shell equispaziate in massa, si ricavano i raggi di bordo r_i(0)
per interpolazione inversa di M(r); Δm_i = M_i - M_{i-1}. La condizione
iniziale NON è esattamente in equilibrio idrostatico rispetto alla EOS
reale (mismatch già noto da Step 1/2) — non è un bug, va documentato nel
codice.

File concordati: `collasso/dynamics.py`, `scripts/step4_demo.py`,
`tests/test_dynamics.py`. Nessuna modifica a `collasso/eos.py`,
`collasso/lane_emden.py`, `collasso/catalog.py` prevista in questo step
(solo riuso via import).

**Nota di revisione**: il reviewer ha esaminato la REV. 1 di questo piano e
ha trovato 4 problemi concreti (2 bloccanti, 2 da chiarire/consigliati),
ora corretti nelle sezioni sotto:
1. **(bloccante)** le masse di test `M_TEST_MSUN`/`M_SUB_CHANDRA_TEST_MSUN`
   della REV. 1 (1.0/0.5 Msun) erano troppo vicine alla regione di
   transizione relativistica: con `M_TEST_MSUN=1.0` la densità centrale
   autoconsistente (n=3/2) dà `rho_c≈4.06e6 g/cm³`, cioè `x_c≈1.28`
   (parametro di Fermi) — **dentro** la regione di transizione, non
   "profondamente non-relativistico" come assunto in REV. 1; a questo `x_c`
   `pressure_chandrasekhar` (EOS esatta) si discosta da
   `pressure_non_relativistic` (usata per costruire l'equilibrio) di
   ~46%, uno scarto **fisico**, non numerico, che invaliderebbe sia il
   test di convergenza (3.5) sia il test sub-Chandrasekhar (3.7) — il
   "residuo" misurato non sarebbe più principalmente errore di
   discretizzazione ma vero squilibrio idrostatico. Corretto riducendo
   `M_TEST_MSUN=0.01`, `M_SUB_CHANDRA_TEST_MSUN=0.005` (`x_c≈0.06`/
   `≈0.0375`, entrambi ben nel regime profondamente non-relativistico),
   con verifica esplicita di `x_c` richiesta al coder — vedi sezione
   costanti sotto e punti 3.1/3.5/3.7.
2. **(bloccante)** `RuntimeError` sollevata dentro `_shell_dvdt` (il RHS
   passato a `solve_ivp`) non viene catturata da scipy — crash non
   gestito del processo, non un errore pulito. Corretto sostituendo
   l'approccio "raise" con: (a) un secondo evento terminale
   `_shell_crossing_event` (stesso pattern di `_collapse_threshold_event`
   e dell'evento di Lane-Emden Step 2), che rileva l'attraversamento di
   shell e ferma l'integrazione in modo pulito; (b) un floor numerico
   fisso `V_FLOOR_CM3=1e-30` applicato al volume di shell dentro il RHS
   (`np.maximum`, mai un'eccezione), per sicurezza fra un passo e l'altro
   del solver. Vedi punti 1.4/1.5/1.6 e nuovo test 3.10.
3. **(da chiarire)** la griglia di Lane-Emden col default `n_points=2000`
   è insufficiente per il test di convergenza a `N=400` shell (~5
   punti/shell, errore di interpolazione che non scala con N).
   Corretto specificando esplicitamente `solve_lane_emden(1.5,
   n_points=N_POINTS_LANE_EMDEN_CONVERGENCE=50000)` (~125 punti/shell a
   N=400) nel test 3.5.
4. **(consigliato)** nessun safeguard sul costo computazionale/step-size
   di `solve_ivp`. Corretto aggiungendo `max_step=t_max_s/2000` esplicito
   in `simulate_collapse`, più una verifica empirica obbligatoria (non
   solo assunta) che `T_MAX_FREE_FALL_MULTIPLIER=10` non produca un
   falso collasso nel caso sub-Chandrasekhar corretto (punto 3.7).
Nessun'altra parte del piano è stata modificata: schema di
discretizzazione (shell Lagrangiane, gravità Newtoniana + pressione da
`pressure_chandrasekhar`), formula di `polytrope_equilibrium_rho_c_gcm3`
(verificata algebricamente dal planner, invariata) e perimetro fisico
dello step (nessuna correzione relativistica, nessun TOV, nessun
neutrini/rotazione/viscosità) restano quelli già validati dal reviewer.

**Nota di revisione REV. 3 (fix mirato al test 3.5, nessuna modifica alla
fisica/discretizzazione)**: il coder ha implementato la REV. 2 alla
lettera — 49/50 test passano; `test_polytrope_equilibrium_initial_acceleration_convergence`
(3.5) fallisce **genuinamente**, non per errore di trascrizione. Diagnosi
confermata dal reviewer e verificata dal planner leggendo direttamente
`collasso/dynamics.py` (`_shell_dvdt`, righe ~231-261): la densità usata
per l'ultima shell (`rho[-1] = delta_m_g[-1]/v[-1]`) è una **media di
volume**, non la densità puntuale al bordo. Per un profilo a superficie
nulla come `theta^n` vicino a `xi1` (n=3/2, `rho ~ s^1.5` con `s` distanza
dalla superficie), questa media introduce un fattore sistematico
`1/(n+1)` che **non scompare aumentando N**. Il reviewer ha derivato
analiticamente il valore limite universale
```
ratio(N->infinito) = |1 - 2/(n+1)^((n+1)/n)|
```
che per n=3/2 dà **ratio_∞ = 0.5656**, in accordo quantitativo eccellente
con i dati empirici del coder (fit con correzione O(N^-0.4) su tutti i
punti N=50..50000). È un limite **strutturale** dello schema di
discretizzazione al bordo libero, non risolvibile infittendo la griglia —
il test 3.5 come scritto in REV. 2 (che si aspetta convergenza a zero su
TUTTE le shell, incluso il bordo) non può mai passare per l'ultima shell,
qualunque N si usi.

Il reviewer ha inoltre segnalato un problema logico separato, indipendente
dal precedente: l'assert di monotonicità con `MONOTONE_SLACK=1.1`
(tolleranza moltiplicativa del 10% per passo) **tollera** una sequenza
sistematicamente **crescente** — con i dati reali del coder (crescenti:
0.49→0.51→0.53→...→0.566) l'assert passa "per sbaglio", perché non
discrimina fra "converge decrescendo con rumore" e "cresce
sistematicamente verso un plateau non nullo".

**Perimetro della correzione**: SOLO la metrica del test 3.5 e la
documentazione (docstring di `dynamics.py`) cambiano. Lo schema di
discretizzazione (`_shell_dvdt`/`_collapse_rhs`), le formule fisiche, il
perimetro dello step (gravità Newtoniana, EOS Chandrasekhar esatta,
nessun neutrini/relativistico/rotazione/viscosità) **restano invariati**,
già validati dal reviewer nella REV. 2 e non rimessi in discussione qui.
Correzioni dettagliate integrate nei punti 1.4bis (nuovo, docstring), 3.1
(nuove costanti di tolleranza, sostituiscono `MONOTONE_SLACK`) e 3.5
(riscrittura della logica di asserzione) sotto.

### Nota di verifica del planner (derivazione controllata prima di consegnarla al coder)

**Verifica della formula chiusa `polytrope_equilibrium_rho_c_gcm3`.** Per
un politropo P=K·ρ^{1+1/n} in equilibrio idrostatico Newtoniano, la
sostituzione standard ρ=ρ_c·θ^n, r=α·ξ nell'equazione di struttura
d/dr(r²/ρ·dP/dr) = -4πG·r²·ρ riporta all'equazione di Lane-Emden con

  α² = (n+1)·K/(4πG) · ρ_c^{1/n - 1}

(derivazione svolta esplicitamente dal planner; per n=3/2 l'esponente vale
1/n-1 = 2/3-1 = -1/3). La massa totale del politropo è

  M = 4π·α³·ρ_c·C_M,  C_M = -ξ1²·θ'(ξ1)

che è **esattamente** la stessa relazione già usata in
`lane_emden.physical_profile` (`massa_g = 4*pi*alpha_cm**3*rho_c_gcm3*
geom_term`, `geom_term = -xi1**2*dtheta_dxi_at_xi1`) — coerenza incrociata
verificata. Sostituendo α³ nella formula della massa ed esplicitando ρ_c
si ottiene, per n=3/2 (n+1=2.5):

  M = 4π·C_M·(2.5·K/(4πG))^{1.5}·ρ_c^{1/2}
  ⟹ ρ_c = ( M / (4π·C_M·(2.5·K/(4πG))^{1.5}) )²

che coincide **esattamente**, termine per termine, con la formula fissata
nell'approccio approvato:
`rho_c = ( M_g / (4*pi*C_M * (2.5*K_NR/(4*pi*G))**1.5) )**2`.
**Verificata algebricamente dal planner, nessun errore trovato — il coder
deve usarla esattamente come scritta, non ri-derivarla.**

**Rischio numerico esplicito da evitare: K_NR ≠ K_UR.** `K` in questa
formula è la costante del limite NON-relativistico (P_NR=K_NR·ρ^{5/3},
n=3/2), NON `k_ultrarelativistic` di `collasso/eos.py` (quella è per n=3,
P_UR∝ρ^{4/3}, usata da `chandrasekhar_mass_msun`). `collasso/eos.py` **non
espone** una funzione `k_non_relativistic(ye)` dedicata — il coder deve
ottenere `k_nr_cgs` con questo metodo, che è **esatto e non
un'approssimazione** (verificato dal planner): dato che
`pressure_non_relativistic(rho_gcm3, ye) = ((3π²)^(2/3)/5)·(ħ²/m_e)·
(ye·rho_gcm3/M_U_G)^(5/3)`, il fattore `rho_gcm3^(5/3)` si fattorizza
esattamente, quindi valutare la funzione a `rho_gcm3=1.0` (in g/cm³,
stesse unità cgs usate ovunque) restituisce **esattamente** `K_NR(ye)`:

  `k_nr_cgs = eos.pressure_non_relativistic(1.0, ye)`

Questo va usato **sempre** al posto di qualunque formula duplicata a mano o
di `k_ultrarelativistic` — il coder deve scrivere un commento esplicito nel
codice (in `tests/test_dynamics.py`, unico punto dove questa costante
serve, vedi punto 3 sotto) che richiama questa nota, per evitare confusione
futura fra i due limiti.

### Vincoli di modello per questo step (da CLAUDE.md, non negoziabili)
- Gravità **solo Newtoniana**: nessuna correzione relativistica sul nucleo
  (responsabilità esclusiva Step 5).
- Nessun limite TOV, nessuna classificazione del remnant (responsabilità
  esclusiva Step 6).
- Nessun termine di raffreddamento neutrini: limite di modello dichiarato
  esplicitamente nel codice e nell'output demo (stesso trattamento delle
  correzioni relativistiche — NON introdotto in questo step).
- Nessuna rotazione/campo magnetico (limiti già esclusi a monte, invariati
  da Step 1).
- Nessuna viscosità artificiale: limite di modello dichiarato
  esplicitamente (possibile ringing numerico in forte compressione),
  accettabile perché questo step si ferma all'innesco del collasso
  (soglia `r_min_frac`), non al bounce.
- EOS usata nella dinamica: **sempre** `pressure_chandrasekhar` (Step 3,
  esatta), mai `pressure_non_relativistic`/`pressure_ultrarelativistic`
  come forza motrice della dinamica (questi ultimi servono SOLO per
  ottenere `K_NR` nel test di equilibrio autoconsistente, vedi nota
  sopra).

### Costanti e parametri numerici fissati dal planner (non a discrezione del coder)
- `R_MIN_FRAC_DEFAULT = 0.1` — soglia di terminazione (10% del raggio
  iniziale della shell più interna).
- `N_POINTS_DEFAULT = 500` — numero di punti di output richiesti a
  `solve_ivp` via `t_eval` (default del parametro `n_points` di
  `simulate_collapse`).
- `T_MAX_FREE_FALL_MULTIPLIER = 10.0` — costante di modulo in
  `dynamics.py`, usata dai chiamanti (demo e test) per calcolare
  `t_max_s = T_MAX_FREE_FALL_MULTIPLIER * free_fall_time_s(rho_c_iniziale)`;
  `simulate_collapse` NON calcola `t_max_s` da sola (parametro
  obbligatorio, nessun default nella firma), per tenere esplicito il
  legame fra la scala temporale scelta e la densità centrale iniziale
  della specifica configurazione.
  - Giustificazione numerica: per `rho_c≈7.5e9 g/cm³` (s20),
    `free_fall_time_s ≈ 24 ms` ⟹ `t_max_s≈240 ms`; per un nucleo
    fortemente instabile (super-Chandrasekhar, nessun supporto di
    pressione/neutrini) ci si attende che la soglia `r_min_frac` sia
    raggiunta entro pochi tempi di caduta libera (1-3), quindi 10 dà
    margine ampio senza costo computazionale eccessivo.
  - **(NUOVO, REV. 2, correzione consigliata 4)** nota di verifica
    obbligatoria per il coder: per il caso sub-Chandrasekhar (test 3.7),
    dopo la correzione delle masse di test (punto 1 sotto) il residuo
    idrostatico iniziale è atteso molto più piccolo (~0.1% invece di
    ~46% della REV. 1), quindi il rischio di falso collasso per accumulo
    di residuo numerico su `10*t_ff` è molto ridotto — ma va comunque
    **verificato empiricamente eseguendo il test**, non assunto.
    Eventuali aggiustamenti vanno fatti SOLO per quel test specifico (non
    il default globale usato da demo/3.6), motivando la scelta finale nel
    report al subagent `critic-fisico` — vedi nota dettagliata al punto
    3.7.
- Integratore `solve_ivp`: `method="RK45"` (stessa scelta di Step 2,
  nessun passaggio a metodi impliciti Radau/BDF senza consultare il
  planner), `rtol=1e-6`, `atol=1e-2` (valori fissati dal planner: gli
  stati hanno scala tipica `r~1e6-1e9 cm`, `v~0-1e9 cm/s`, quindi
  `atol=1e-2` è trascurabile rispetto a qualunque valore fisicamente
  rilevante, e `rtol` domina l'integrazione).
  - **(NUOVO, REV. 2, correzione consigliata 4)** `max_step = t_max_s /
    2000.0`, passato esplicitamente a `solve_ivp` in `simulate_collapse`
    (non lasciato al default automatico dello step-size di RK45):
    garantisce almeno 2000 passi massimi anche in caso di rigidità
    numerica vicino al collasso, evitando che il solver riduca lo step
    fino al blocco silenzioso del ciclo senza mai restituire un errore
    chiaro.
- `V_FLOOR_CM3 = 1e-30` — **(NUOVO, REV. 2, correzione bloccante 2)**
  floor positivo fisso (cm^3) applicato al volume di ciascuna shell
  dentro `_shell_dvdt` (`V = np.maximum(V_raw, V_FLOOR_CM3)`), per evitare
  NaN/divisioni per zero SENZA mai sollevare un'eccezione dentro il RHS
  passato a `solve_ivp` (scipy non la catturerebbe — vedi nota di
  revisione in testa alla sezione). Sostituisce il precedente
  `RuntimeError` sollevato dentro `_shell_dvdt` nella REV. 1; non è il
  meccanismo che rileva/segnala lo shell crossing (quello è l'evento
  terminale `_shell_crossing_event`, punto 1.4), è solo una rete di
  sicurezza numerica residua.
- `N_SHELLS_DEMO_DEFAULT = 200` — numero di shell usato in
  `scripts/step4_demo.py` (bilancio risoluzione/tempo di esecuzione:
  sistema ODE 2·200=400 componenti).
- `N_SHELLS_TEST_DEFAULT = 50` — numero di shell usato nei test di
  innesco/non-innesco del collasso (`test_dynamics.py`), per velocità di
  esecuzione della suite.
- `N_CONVERGENCE_LIST = [50, 100, 200, 400]` — valori di N per il test di
  convergenza dell'equilibrio autoconsistente (progressione geometrica ×2).
- `N_POINTS_LANE_EMDEN_CONVERGENCE = 50000` — **(NUOVO, REV. 2, correzione
  da chiarire 3)** numero di punti della griglia xi passato esplicitamente
  a `solve_lane_emden(1.5, n_points=N_POINTS_LANE_EMDEN_CONVERGENCE)` nel
  test di convergenza (3.5) — NON il default `n_points=2000` di
  `solve_lane_emden` (Step 2), che a `N=400` shell darebbe solo ~5
  punti/shell per l'interpolazione inversa `M(r)->r` di
  `build_initial_shells`, rischiando un errore di interpolazione che non
  scala con N e appiattisce la convergenza attesa. Con 50000 punti si
  hanno ~125 punti/shell anche a `N=400`, abbondantemente sufficienti.
- `M_TEST_MSUN = 0.01` — **(CORRETTO, REV. 2, correzione bloccante 1 — era
  1.0 in REV. 1)** massa del politropo n=3/2 autoconsistente usato nel
  test di convergenza (scelta indipendente dal catalogo, per non
  mescolare il noto mismatch dei 4 parametri placeholder — Step 1/2 —
  con la verifica di convergenza del solo schema numerico). `YE_TEST =
  0.5` (invariato).
  - **Motivazione della correzione (dal reviewer, verificata dal
    planner)**: con il valore precedente `M_TEST_MSUN=1.0`, la densità
    centrale autoconsistente (formula chiusa n=3/2,
    `polytrope_equilibrium_rho_c_gcm3`) risulta `rho_c≈4.06e6 g/cm³`,
    corrispondente a `x_c≈1.28` (parametro di Fermi) — **dentro** la
    regione di transizione relativistica, NON "profondamente
    non-relativistico" come erroneamente assunto nel piano REV. 1: qui
    `pressure_chandrasekhar` (EOS esatta) si discosta da
    `pressure_non_relativistic` (usata per costruire l'equilibrio) di
    ~46%. Questo scarto è **fisico**, non numerico — invaliderebbe sia il
    test di convergenza (3.5) sia il test sub-Chandrasekhar (3.7), perché
    il "residuo" misurato non sarebbe più principalmente errore di
    discretizzazione ma vero squilibrio idrostatico.
  - Dato che per n=3/2 `rho_c ∝ M²` (dalla formula chiusa) e quindi
    `x_c ∝ M^(2/3)`, per portare `x_c` a un valore sicuro (~0.06, dove lo
    scarto `f_esatta/f_leading-1 ≈ -(5/14)x²` è solo ~0.13%, un ordine di
    grandezza sotto `RATIO_TOL_N400=1e-2`) serve `M_TEST_MSUN≈0.01 Msun`
    (non più 1.0) — valore ora fissato sopra.
  - Verifica esplicita richiesta al coder (vedi anche 3.5): stampare/
    asserire `x_c` come parte del test, non fidarsi solo del calcolo del
    planner.
- `M_SUB_CHANDRA_TEST_MSUN = 0.005` — **(CORRETTO, REV. 2, correzione
  bloccante 1 — era 0.5 in REV. 1)** massa del caso sub-Chandrasekhar
  "costruito ad hoc" (stesso politropo n=3/2 autoconsistente, `Ye=0.5`;
  `chandrasekhar_mass_msun(0.5)≈1.4559 Msun` da Step 3, quindi 0.005 Msun
  è ≈1/290 di M_Ch, margine ancora più ampio del precedente ≈1/3). Con la
  stessa scalatura `x_c ∝ M^(2/3)` della nota sopra, questa massa dà
  `x_c≈0.0375`, ANCORA più profondamente non-relativistica del caso
  `M_TEST_MSUN=0.01` — questa volta l'affermazione "profondamente
  non-relativistica" è verificata numericamente (a differenza del piano
  REV. 1, dove la stessa dicitura era usata per `M_SUB_CHANDRA_TEST_MSUN=
  0.5`, che in realtà dava `x_c` nella stessa regione di transizione del
  caso da 1.0 Msun — stesso equivoco corretto qui, non un problema
  separato).

### 0. `collasso/constants.py`
Nessuna modifica necessaria in questo step: tutte le costanti richieste
(`G_CGS`, `M_SUN_G`) sono già presenti da Step 2/3.

### 1. Modulo dinamica — `collasso/dynamics.py`

1.1. `free_fall_time_s(rho_gcm3)`: `t_ff = sqrt(3*pi/(32*G_CGS*rho_gcm3))`
     (formula standard del tempo di caduta libera per una sfera uniforme
     di densità `rho_gcm3`). Vettorizzata (numpy). Guardia: `rho_gcm3 > 0`
     stretto → `ValueError` altrimenti (stesso stile di
     `collasso.eos._validate_rho`).

1.2. `build_initial_shells(r_cm, rho_gcm3, n_shells) -> (delta_m_g,
     m_enclosed_g, r0_cm)`: algoritmo fissato dal planner (esatto, non a
     discrezione del coder):
     - Validare `n_shells >= 2` (`ValueError` altrimenti),
       `len(r_cm) == len(rho_gcm3)`, `rho_gcm3` tutto `>0`, `r_cm`
       strettamente crescente.
     - `M_r = scipy.integrate.cumulative_trapezoid(4*np.pi*r_cm**2*
       rho_gcm3, r_cm, initial=0.0)` (array stessa lunghezza di `r_cm`,
       `M_r[0]=0` per convenzione — nota: `r_cm[0]` non è esattamente 0
       per costruzione di `physical_profile` [xi0=1e-4], quindi si
       trascura una quantità di massa trascurabile fra r=0 e r_cm[0],
       stessa approssimazione già accettata da Lane-Emden dallo Step 2,
       non un nuovo limite).
     - `M_total = M_r[-1]`.
     - `M_targets = np.linspace(M_total/n_shells, M_total, n_shells)` (N
       shell equispaziate in massa, da Δm a M_total).
     - `r0_cm = np.interp(M_targets, M_r, r_cm)` (interpolazione inversa;
       valida perché `M_r` è strettamente crescente dato `rho_gcm3>0`).
     - `delta_m_g = np.diff(np.concatenate(([0.0], M_targets)))` (per
       costruzione, essendo `M_targets` equispaziato, tutti gli elementi
       sono uguali a `M_total/n_shells`; usare `np.diff` comunque, per
       chiarezza e robustezza anche se in futuro la spaziatura in massa
       cambiasse).
     - `m_enclosed_g = M_targets` (uguale, per costruzione, a
       `np.cumsum(delta_m_g)`).

1.3. `polytrope_equilibrium_rho_c_gcm3(k_nr_cgs, m_g, xi1,
     dtheta_dxi_at_xi1)`: formula chiusa **verificata dal planner** (vedi
     nota sopra), da usare esattamente come scritta:
     - `geom_term = -xi1**2 * dtheta_dxi_at_xi1` (stesso nome/segno di
       `lane_emden.physical_profile`); guardia `geom_term > 0` altrimenti
       `ValueError` (stesso pattern difensivo di Step 2).
     - Guardie aggiuntive: `k_nr_cgs > 0`, `m_g > 0`, altrimenti
       `ValueError`.
     - `rho_c_gcm3 = ( m_g / (4*np.pi*geom_term*(2.5*k_nr_cgs/
       (4*np.pi*G_CGS))**1.5) )**2`.
     - Nota nel docstring: questa funzione è specifica del caso **n=3/2**
       (l'esponente 2.5=n+1 e la potenza 1.5 nella formula sono validi
       solo per n=3/2); NON generalizzare a n arbitrario senza consultare
       il planner.

1.4. Funzioni private (nomi fissati, stesso stile di
     `_lane_emden_rhs`/`_theta_zero_event` di Step 2):
     - `_shell_dvdt(r_cm, m_enclosed_g, delta_m_g, ye) -> np.ndarray`
       (lunghezza N): calcola le accelerazioni di TUTTE le shell,
       **vettorizzata (nessun loop Python esplicito sulle shell)**,
       algoritmo fissato dal planner:
       - `r_full = np.concatenate(([0.0], r_cm))` (N+1 elementi, r_0=0
         sempre incluso)
       - `V_raw = (4.0/3.0)*np.pi*(r_full[1:]**3 - r_full[:-1]**3)` (N
         elementi)
       - `V = np.maximum(V_raw, V_FLOOR_CM3)` — **(REV. 2, correzione
         bloccante 2, sostituisce la guardia `RuntimeError` della REV.
         1)**: floor positivo fisso, MAI un'eccezione. Motivazione:
         un'eccezione sollevata dentro il RHS passato a `solve_ivp` non
         viene catturata da scipy e causa un crash non gestito
         dell'intero processo, invece di una terminazione pulita
         dell'integrazione. Il floor evita NaN/divisioni per zero nel
         caso `solve_ivp` valuti `fun` su stati con volumi di shell
         nulli/negativi fra un passo e l'altro (prima che l'evento
         terminale sotto abbia la possibilità di fermare
         l'integrazione); NON è il meccanismo che rileva/segnala lo
         shell crossing come esito — quello è l'evento terminale
         `_shell_crossing_event` (nuovo, vedi sotto), che in condizioni
         normali ferma l'integrazione PRIMA che il floor debba
         intervenire.
       - `rho = delta_m_g / V`
       - `P = collasso.eos.pressure_chandrasekhar(rho, ye)` (una sola
         chiamata vettorizzata)
       - `grav = -G_CGS * m_enclosed_g / r_cm**2` (N elementi, stessa
         formula per bordo interno ed esterno)
       - termine di pressione, interior (indici 0..N-2 in notazione
         0-based, corrispondenti a i=1..N-1): `press_int =
         -4*np.pi*r_cm[:-1]**2*(P[1:]-P[:-1]) /
         ((delta_m_g[:-1]+delta_m_g[1:])/2.0)`
       - termine di pressione, bordo esterno (indice N-1, i=N):
         `press_ext = 4*np.pi*r_cm[-1]**2*P[-1] / (delta_m_g[-1]/2.0)`
       - `dvdt = grav.copy(); dvdt[:-1] += press_int; dvdt[-1] +=
         press_ext; return dvdt`
     - `_collapse_rhs(t, y, m_enclosed_g, delta_m_g, ye, n) ->
       np.ndarray`: `r = y[:n]`, `v = y[n:]`, ritorna
       `np.concatenate([v, _shell_dvdt(r, m_enclosed_g, delta_m_g,
       ye)])`. Non solleva mai eccezioni per condizioni fisiche/numeriche
       (solo `_shell_dvdt` con il floor sopra) — coerente con la
       correzione bloccante 2.
     - `_collapse_threshold_event(t, y, r_min_cm) -> float`: ritorna
       `y[0] - r_min_cm` (attraversamento della shell più interna sotto
       la soglia); `direction=-1`, `terminal=True` (stesso pattern di
       `_theta_zero_event`, Step 2).
     - `_shell_crossing_event(t, y, n) -> float`: **(NUOVO, REV. 2,
       correzione bloccante 2)** evento terminale aggiuntivo che rileva
       l'attraversamento di shell (raggi di bordo consecutivi che si
       incrociano o collassano a spessore nullo, incluso il centro):
       `r_full = np.concatenate(([0.0], y[:n]))`; `gaps =
       np.diff(r_full)`; ritorna `np.min(gaps)`. `direction=-1`,
       `terminal=True` (stesso pattern di `_collapse_threshold_event` e
       di `_theta_zero_event`, Step 2). Se questo evento scatta prima di
       `_collapse_threshold_event`, l'integrazione termina comunque in
       modo pulito, senza mai sollevare un'eccezione fuori da
       `solve_ivp` — trattato come `collapsed=True` in
       `CollapseSolution`, con `collapse_reason="shell_crossing"` (vedi
       1.5), per distinguerlo esplicitamente dal raggiungimento della
       soglia centrale (`collapse_reason="r_min_threshold"`). Nota di
       modello: uno shell crossing è un sintomo diretto del limite già
       dichiarato "nessuna viscosità artificiale" (possibile ringing
       numerico in forte compressione) — non un errore di
       programmazione, va trattato come esito fisico plausibile del
       modello semplificato, non nascosto.

1.4bis. **(NUOVO, REV. 3)** Docstring esplicito, obbligatorio, nello
     stesso stile delle dichiarazioni di limite già presenti nel modulo
     (neutrini, viscosità, rotazione — vedi docstring di modulo, righe
     14-24) da aggiungere al docstring di modulo E/O al docstring di
     `_shell_dvdt` (a discrezione del coder dove risulti più leggibile,
     purché presente in almeno uno dei due, preferibilmente entrambi con
     un rimando incrociato): dichiarare esplicitamente che la
     discretizzazione "densità media di shell (`delta_m/V`) + differenza
     in avanti verso P=0 al bordo libero" **non converge a zero
     sull'ultima shell** per un politropo con densità nulla in superficie
     (comportamento a legge di potenza, `rho ~ s^n` vicino al bordo, `s`
     distanza dalla superficie) — l'errore converge invece a un valore
     limite non nullo, universale nella forma
     `|1 - 2/(n+1)^((n+1)/n)|` (derivato dal reviewer, confermato dal
     planner; per n=3/2 vale ≈0.5656). Va precisato che si tratta di un
     limite **strutturale** dello schema (non risolvibile aumentando N)
     ma **confinato a un numero fisso e piccolo di shell** vicino al
     bordo (frazione di massa trascurabile, O(1/N) per shell) — non
     influenza la dinamica macroscopica del collasso, che dipende dalle
     shell interne (dove lo schema converge correttamente) e dalla massa
     totale/gravità, non dal dettaglio locale di pressione all'ultima
     shell.

1.5. `CollapseSolution` (dataclass): campi `t_s: np.ndarray` (lunghezza
     `T`), `r_cm: np.ndarray` (shape `(N, T)`, riga i = traiettoria della
     shell i — stessa convenzione di `solve_ivp.y`, NESSUNA
     trasposizione), `v_cms: np.ndarray` (shape `(N, T)`, stessa
     convenzione), `rho_c_gcm3_t: np.ndarray` (lunghezza `T`, densità
     della shell 1: `delta_m_g[0] / ((4/3)*pi*r_cm[0,:]**3)`),
     `collapsed: bool`, `t_collapse_s: float | None`,
     `collapse_reason: str | None` — **(NUOVO, REV. 2, correzione
     bloccante 2)** valori ammessi: `"r_min_threshold"` (evento 1,
     soglia centrale raggiunta), `"shell_crossing"` (evento 2, shell
     collassate/incrociate), `None` se `collapsed is False`. Docstring
     deve dichiarare esplicitamente la convenzione di shape (righe=shell,
     colonne=tempo) per evitare bug di trasposizione a valle (Step 7,
     animazione), e i tre valori ammessi di `collapse_reason`.

1.6. `simulate_collapse(delta_m_g, r0_cm, ye, t_max_s,
     r_min_frac=R_MIN_FRAC_DEFAULT, n_points=N_POINTS_DEFAULT) ->
     CollapseSolution`:
     - Validare: `t_max_s > 0`, `0 < r_min_frac < 1`,
       `len(delta_m_g) == len(r0_cm) >= 2`, `delta_m_g` tutto `>0`,
       `r0_cm` strettamente crescente (altrimenti `ValueError`).
     - `n = len(r0_cm)`; `m_enclosed_g = np.cumsum(delta_m_g)`.
     - `y0 = np.concatenate([r0_cm, np.zeros(n)])` (shell inizialmente a
       riposo, come da approccio approvato).
     - `r_min_cm = r_min_frac * r0_cm[0]`.
     - evento 1 (soglia centrale) = `_collapse_threshold_event` con
       `args=(r_min_cm,)`, `direction=-1`, `terminal=True`.
     - evento 2 **(NUOVO, REV. 2, correzione bloccante 2)** (shell
       crossing) = `_shell_crossing_event` con `args=(n,)`,
       `direction=-1`, `terminal=True`. Passare `events=[evento1,
       evento2]` (lista, non un singolo evento) a `solve_ivp` — sostituisce
       il precedente approccio "raise `RuntimeError` dentro il RHS" della
       REV. 1, che scipy non cattura e causerebbe un crash non gestito
       (vedi nota di revisione in testa alla sezione).
     - `max_step = t_max_s / 2000.0` **(NUOVO, REV. 2, correzione
       consigliata 4)**: passato esplicitamente a `solve_ivp`, garantisce
       almeno 2000 passi massimi anche in caso di rigidità numerica vicino
       al collasso, evitando che il solver riduca silenziosamente lo step
       fino al blocco del ciclo senza mai restituire un errore chiaro.
     - `t_eval = np.linspace(0.0, t_max_s, n_points)`.
     - `sol = solve_ivp(fun=_collapse_rhs, t_span=(0.0, t_max_s),
       y0=y0, method="RK45", rtol=1e-6, atol=1e-2, max_step=max_step,
       t_eval=t_eval, events=[evento1, evento2],
       args=(m_enclosed_g, delta_m_g, ye, n))`.
     - Check esplicito `sol.success` → `RuntimeError` con `sol.message`
       se `False` (stesso pattern obbligatorio di Step 2, punto 1.5 del
       piano Step 2), PRIMA di processare gli eventi o costruire
       `CollapseSolution`.
     - **(REV. 2)** determinazione di `collapsed`/`t_collapse_s`/
       `collapse_reason` a partire da `sol.t_events` (ora una lista di
       due array, uno per evento, nello stesso ordine di `events=[...]`):
       - `threshold_hit = len(sol.t_events[0]) > 0`;
         `crossing_hit = len(sol.t_events[1]) > 0`;
       - `collapsed = threshold_hit or crossing_hit`;
       - se nessuno dei due: `t_collapse_s = None`,
         `collapse_reason = None`;
       - se solo `threshold_hit`: `t_collapse_s =
         float(sol.t_events[0][0])`, `collapse_reason =
         "r_min_threshold"`;
       - se solo `crossing_hit`: `t_collapse_s =
         float(sol.t_events[1][0])`, `collapse_reason =
         "shell_crossing"`;
       - se entrambi (caso limite, eventi simultanei nello stesso passo
         del solver): scegliere il tempo minimo fra i due e il
         `collapse_reason` corrispondente; se coincidono esattamente,
         dare priorità a `"r_min_threshold"` (convenzione arbitraria ma
         deterministica, da documentare nel docstring — caso atteso
         raro/mai osservato in pratica);
       - mai dedotto dall'ultimo punto di `t_eval`, sempre dal tempo
         esatto dell'evento che ha terminato l'integrazione (stesso
         principio di `xi1` in Step 2).
     - `r_cm = sol.y[:n, :]`, `v_cms = sol.y[n:, :]`, `t_s = sol.t`.
     - `rho_c_gcm3_t = delta_m_g[0] / ((4.0/3.0)*np.pi*r_cm[0, :]**3)`
       (calcolato come post-processing, non durante l'integrazione).
     - Nota nel docstring: se un evento terminale scatta prima di
       `t_max_s`, `scipy.integrate.solve_ivp` restringe automaticamente
       `t_eval`/`sol.t`/`sol.y` ai soli punti antecedenti l'evento —
       comportamento nativo di scipy, non richiede gestione manuale
       aggiuntiva nel codice, ma va documentato per chiarezza.

1.7. Nota di performance (obbligatoria nel docstring del modulo):
     `_shell_dvdt` deve restare vettorizzata (nessun loop Python esplicito
     su `range(n)`) — con `N_SHELLS_DEMO_DEFAULT=200` e
     `T_MAX_FREE_FALL_MULTIPLIER=10`, un'implementazione a loop esplicito
     rischia tempi di esecuzione eccessivi in `scripts/step4_demo.py`. Se,
     nonostante la vettorizzazione, il tempo di esecuzione del demo
     risultasse comunque eccessivo (>1-2 minuti), il coder deve segnalarlo
     esplicitamente nell'output/commit (non ridurre silenziosamente
     `N_SHELLS_DEMO_DEFAULT` senza annotarlo). Il floor `V_FLOOR_CM3`
     (punto 1.4) è una rete di sicurezza numerica, non un sostituto della
     vettorizzazione né dell'evento di shell crossing.

### 2. Script demo — `scripts/step4_demo.py`

2.1. Caricare il catalogo di riferimento, selezionare `s20`
     (`get_progenitor_by_id(catalog, "s20")`).
2.2. Confrontare `massa_nucleo_msun` (1.50) con
     `chandrasekhar_mass_msun(0.46)` (Step 3, ≈1.2323 Msun): stampare
     esplicitamente che il nucleo è **supra-Chandrasekhar** (scarto
     percentuale `100*(1.50-1.2323)/1.2323`).
2.3. Costruire la condizione iniziale: `solve_lane_emden(n=2.0)` (n del
     catalogo per s20), `physical_profile(solution, densita_centrale_gcm3,
     massa_nucleo_msun)` (Step 2, coppia primaria come da convenzione già
     stabilita), poi `build_initial_shells(profile.r_cm, profile.rho_gcm3,
     N_SHELLS_DEMO_DEFAULT)`.
2.4. Calcolare `t_max_s = T_MAX_FREE_FALL_MULTIPLIER *
     free_fall_time_s(densita_centrale_gcm3)`; stampare il tempo di
     caduta libera in ms.
2.5. Eseguire `simulate_collapse(delta_m_g, r0_cm, ye=0.46,
     t_max_s=t_max_s)` (parametri di default per `r_min_frac`/
     `n_points`).
2.6. Stampare: densità centrale iniziale (`rho_c_gcm3_t[0]`) e finale
     (`rho_c_gcm3_t[-1]`), `collapsed`, `t_collapse_s` (se disponibile),
     `collapse_reason` (se disponibile — **NUOVO, REV. 2**, campo
     aggiunto a `CollapseSolution`, non va lasciato silenziosamente fuori
     dall'output) e il suo rapporto con `t_max_s`/tempo di caduta libera.
2.7. Riga di disclaimer **obbligatoria** (stesso principio delle righe di
     disclaimer di Step 2/3): dichiarare esplicitamente i limiti di
     questo step — nessun termine neutrini, nessuna correzione
     relativistica (Newtoniana pura), nessuna viscosità artificiale,
     condizione iniziale non esattamente in equilibrio idrostatico
     rispetto alla EOS reale (mismatch noto dal catalogo placeholder,
     Step 1/2), integrazione fermata all'innesco del collasso
     (`r_min_frac` o shell crossing), non al bounce.
2.8. **Nessun plot Matplotlib** (riservato a Step 7). Nessun output
     verboso: solo le righe/riepiloghi sopra elencati (regola CLAUDE.md
     sul filtraggio output).

### 3. Test — `tests/test_dynamics.py`

3.1. Costanti di tolleranza in testa al file (fissate dal planner):
     - `TOL_MASS_SHELLS_INTERNAL_RTOL = 1e-8` — `sum(delta_m_g)` vs
       `M_r[-1]` (totale calcolato dalla STESSA integrazione trapezoidale
       usata per costruire le shell): identità algebrica per telescopio
       di somma, tolleranza vicina alla precisione macchina.
     - `TOL_MASS_SHELLS_EXTERNAL_RTOL = 1e-3` — `sum(delta_m_g)` vs
       `massa_nucleo_msun*M_SUN_G` (valore nominale passato a
       `physical_profile`): tolleranza più larga perché include l'errore
       di quadratura trapezoidale sulla griglia finita di Lane-Emden
       (2000 punti) rispetto alla relazione massa-raggio analitica — non
       un errore di costruzione delle shell.
     - `TOL_R_LAST_RTOL = 1e-3` — `r0_cm[-1]` vs `profile.R_cm`, stessa
       motivazione di quadratura.
     - `RATIO_TOL_N400 = 1e-2` — soglia massima ammessa per il rapporto
       `|accelerazione netta|/|accelerazione gravitazionale|` a `N=400`
       nel test di convergenza dell'equilibrio autoconsistente (applicata
       ora, REV. 3, solo alle shell interne — vedi `K_BOUNDARY_EXCLUDE`
       sotto).
     - `RATIO_REDUCTION_FACTOR_MIN = 4.0` — riduzione minima attesa del
       rapporto massimo fra `N=50` e `N=400` (fattore 8× in N): 4× è un
       limite inferiore prudente e non-fragile, soddisfatto sia da una
       convergenza `O(1/N)` (atteso ≈8×) sia da `O(1/N²)` (atteso ≈64×)
       anche in presenza di rumore/effetti di bordo che rallentano la
       convergenza rispetto al caso ideale.
     - `MONOTONE_SLACK = 1.1` — **(SUPERSEDUTA, REV. 3, vedi nota di
       revisione in testa alla sezione)** questa tolleranza moltiplicativa
       del 10% per passo, applicata al rapporto calcolato su TUTTE le
       shell (incluso il bordo esterno), tollera una sequenza
       sistematicamente crescente (verificato sui dati reali del coder:
       0.49→0.51→...→0.566, ~2-4%/raddoppio di N) senza discriminarla da
       una vera convergenza rumorosa — problema logico separato dal
       limite strutturale del bordo, segnalato dal reviewer. Costante
       mantenuta qui SOLO per tracciabilità storica, NON più usata dal
       test 3.5 dopo la correzione REV. 3 (vedi nuove costanti sotto).
     - `K_BOUNDARY_EXCLUDE` — **(NUOVO, REV. 3)** numero FISSO e piccolo
       (non scalante con N) di shell più esterne escluse dal calcolo del
       residuo di convergenza principale in 3.5, per isolare il limite
       strutturale del bordo libero (vedi nota di revisione) dalla
       convergenza reale delle shell interne. Valore NON fissato a
       priori dal planner in questo caso specifico: il coder deve
       determinarlo empiricamente, provando `k=1` (esclude solo l'ultima
       shell) e, se insufficiente, `k=2` (esclude le ultime due),
       calcolando `ratios_interior = max(|a_net[:-k]|/a_grav[:-k])` su
       `N_CONVERGENCE_LIST=[50,100,200,400]` per ciascun k candidato, e
       fissando nel codice il valore MINIMO di k che dà una convergenza
       pulita (sequenza non-crescente entro `MONOTONE_TOL_RTOL` sotto,
       `ratios_interior[-1] < RATIO_TOL_N400`,
       `ratios_interior[0]/ratios_interior[-1] >
       RATIO_REDUCTION_FACTOR_MIN`). Il valore scelto va accompagnato da
       un commento nel codice che riporta i risultati empirici osservati
       per k=1 (e k=2 se provato) e la motivazione della scelta finale —
       non un numero "magico" senza giustificazione tracciata.
     - `RATIO_BOUNDARY_SANITY_MAX = 1.0` — **(NUOVO, REV. 3)** soglia
       debole per l'assert separato sulle `K_BOUNDARY_EXCLUDE` shell di
       bordo escluse dal residuo di convergenza: verifica solo che il
       rapporto |a_net|/a_grav su quelle shell resti `< 1.0` (cioè non
       diverga/non esploda) per ogni N in `N_CONVERGENCE_LIST`, SENZA
       pretendere che converga a zero — documenta il limite strutturale
       come fatto testato esplicitamente (valore di riferimento, non
       vincolante per l'assert ma da riportare in un commento: plateau
       teorico `ratio_∞≈0.5656` per n=3/2, derivato dal reviewer,
       ampiamente sotto 1.0).
     - `MONOTONE_TOL_RTOL` — **(NUOVO, REV. 3)** tolleranza relativa
       stretta (in sostituzione della `MONOTONE_SLACK=1.1` superseduta)
       per l'assert di monotonicità applicato SOLO a `ratios_interior`
       (shell interne, dopo l'esclusione di `K_BOUNDARY_EXCLUDE`): il
       rapporto massimo a `N` più grande deve essere `<=
       (1+MONOTONE_TOL_RTOL) *` il rapporto a `N` precedente. Valore
       numerico da determinare empiricamente dal coder (NON dal planner a
       priori, perché dipende dalla scala di rumore osservata sui dati
       reali dopo l'esclusione del bordo): deve essere abbastanza stretto
       da NON tollerare una crescita sistematica come quella osservata
       nella REV. 2 (~2-4% per raddoppio di N sull'intera sequenza, bordo
       incluso) ma abbastanza largo da assorbire il rumore numerico
       genuino delle shell interne (residuo mediano osservato ~0.001,
       "piatto"). Punto di partenza suggerito per la calibrazione:
       provare valori nella fascia 0.005-0.02 (0.5%-2% per passo) e
       verificare empiricamente quale discrimina correttamente sui 4
       punti di `N_CONVERGENCE_LIST`; il valore finale scelto va
       documentato con un commento che riporta la sequenza
       `ratios_interior` osservata e la motivazione della soglia.
     - `X_C_SANITY_MAX = 0.1` — **(NUOVO, REV. 2, correzione bloccante
       1)** soglia di sicurezza per il parametro di Fermi al centro delle
       configurazioni di equilibrio autoconsistenti usate nei test 3.5 e
       3.7: `x_c = eos.fermi_x(rho_c, YE_TEST) < X_C_SANITY_MAX` garantisce
       che lo scarto fra `pressure_chandrasekhar` (EOS esatta) e
       `pressure_non_relativistic` (usata per costruire l'equilibrio;
       scarto leading-order atteso `≈-(5/14)x²`) resti sotto ~0.3%, un
       ordine di grandezza sotto `RATIO_TOL_N400=1e-2` — coerente con la
       correzione bloccante 1 (masse di test ridotte a
       `M_TEST_MSUN=0.01`/`M_SUB_CHANDRA_TEST_MSUN=0.005`, attese
       `x_c≈0.06`/`x_c≈0.0375`, entrambe ben sotto `X_C_SANITY_MAX`). Il
       test deve asserire questa condizione esplicitamente, non solo
       fidarsi del calcolo del planner nel piano.
3.2. `test_build_initial_shells_mass_conservation_internal`: usa il
     profilo fisico di s20 (Step 2, `n=2.0`) con `n_shells=100`; verifica
     `abs(sum(delta_m_g) - M_r_totale)/M_r_totale <
     TOL_MASS_SHELLS_INTERNAL_RTOL` (M_r_totale ricalcolato nel test con
     la stessa formula di trapezio, per coerenza — non importato come
     privato dal modulo, per essere un controllo di conservazione
     realmente indipendente dall'implementazione interna specifica).
3.3. `test_build_initial_shells_mass_conservation_external`: stesso caso,
     verifica `abs(sum(delta_m_g) - massa_nucleo_msun*M_SUN_G)/
     (massa_nucleo_msun*M_SUN_G) < TOL_MASS_SHELLS_EXTERNAL_RTOL`.
3.4. `test_build_initial_shells_radii_monotonic_and_surface_match`:
     `r0_cm` strettamente crescente (`np.all(np.diff(r0_cm) > 0)`);
     `abs(r0_cm[-1] - profile.R_cm)/profile.R_cm < TOL_R_LAST_RTOL`;
     `m_enclosed_g` uguale a `np.cumsum(delta_m_g)` entro `rtol=1e-12`.
3.5. `test_polytrope_equilibrium_initial_acceleration_convergence`:
     **(REV. 2, correzione bloccante 1 + da chiarire 3; REV. 3, correzione
     della metrica di convergenza — vedi nota di revisione in testa alla
     sezione)** `solution = solve_lane_emden(1.5,
     n_points=N_POINTS_LANE_EMDEN_CONVERGENCE)` (50000 punti, esplicito —
     NON il default 2000 di `solve_lane_emden`: a `N=400` shell, 2000
     punti darebbero solo ~5 punti/shell per l'interpolazione inversa
     `M(r)->r` di `build_initial_shells`, rischiando un errore di
     interpolazione che non scala con N e appiattisce la convergenza; con
     50000 punti si hanno ~125 punti/shell anche a N=400, abbondantemente
     sufficienti). La stessa soluzione va calcolata una sola volta, fuori
     dal loop su `N` (la soluzione Lane-Emden non dipende da `N`, è più
     veloce e non introduce alcuna differenza nel confronto). Per ogni `N`
     in `N_CONVERGENCE_LIST`:
     - `k_nr_cgs = eos.pressure_non_relativistic(1.0, YE_TEST)` (commento
       esplicito nel test che richiama la nota planner su K_NR vs K_UR,
       vedi sopra);
     - `rho_c = polytrope_equilibrium_rho_c_gcm3(k_nr_cgs,
       M_TEST_MSUN*M_SUN_G, solution.xi1, solution.dtheta_dxi_at_xi1)`
       (con `M_TEST_MSUN=0.01`, vedi correzione bloccante 1 nella sezione
       costanti sopra);
     - **(NUOVO, REV. 2, correzione bloccante 1)** `x_c =
       eos.fermi_x(rho_c, YE_TEST)`; il test deve stampare/includere
       `x_c` in un messaggio di assert (non silenziato) e asserire
       esplicitamente `x_c < X_C_SANITY_MAX` — verifica che la
       configurazione di test sia realmente nel regime profondamente
       non-relativistico assunto dall'equilibrio costruito con
       `pressure_non_relativistic`, non solo dichiarato a parole nel
       piano;
     - `profile = physical_profile(solution, rho_c, M_TEST_MSUN)`;
     - `delta_m_g, m_enclosed_g, r0_cm = build_initial_shells(
       profile.r_cm, profile.rho_gcm3, N)`;
     - `a_net = dynamics._shell_dvdt(r0_cm, m_enclosed_g, delta_m_g,
       YE_TEST)` (import esplicito del privato, stesso precedente di
       `_gamma1_of_x`, Step 3 — commento nel test che dichiara che questo
       è un controllo di **convergenza della discretizzazione**, non una
       validazione fisica indipendente della formula di accelerazione,
       stesso principio del punto 3.7 di Step 2);
     - `a_grav = G_CGS*m_enclosed_g/r0_cm**2`;
     - **(MODIFICATO, REV. 3 — vedi nota di revisione in testa alla
       sezione e nuove costanti in 3.1)** `ratio_full =
       np.abs(a_net)/a_grav` (array di lunghezza N, tutte le shell);
       separare in `ratio_interior = ratio_full[:-K_BOUNDARY_EXCLUDE]` e
       `ratio_boundary = ratio_full[-K_BOUNDARY_EXCLUDE:]`;
       `ratio_interior_max = np.max(ratio_interior)`;
     - raccogliere `ratio_interior_max` per ogni `N`, poi asserire (sugli
       stessi 4 punti di `N_CONVERGENCE_LIST`, stessa logica della REV. 2
       ma ristretta alle shell interne):
       - sequenza `ratio_interior_max` non-crescente entro
         `MONOTONE_TOL_RTOL` (sostituisce `MONOTONE_SLACK`, superseduta —
         tolleranza stretta, deve effettivamente discriminare rumore da
         trend crescente sistematico, non solo "passare");
       - `ratio_interior_max` a `N=400 < RATIO_TOL_N400`;
       - `ratio_interior_max(N=50)/ratio_interior_max(N=400) >
         RATIO_REDUCTION_FACTOR_MIN`;
     - **(NUOVO, REV. 3)** assert SEPARATO e più debole sulle shell di
       bordo escluse, per ogni N in `N_CONVERGENCE_LIST`:
       `assert np.all(ratio_boundary < RATIO_BOUNDARY_SANITY_MAX)` — NON
       pretende convergenza a zero, verifica solo che il rapporto non
       diverga; commento obbligatorio nel test che cita il valore limite
       teorico `ratio_∞≈0.5656` (n=3/2, derivato dal reviewer) e rimanda
       alla nota di limite strutturale nel docstring di `dynamics.py`
       (punto 1.4bis) — così il limite è documentato come fatto testato
       esplicitamente, non solo descritto in prosa nel piano/report.
     - **(NUOVO, REV. 3, non bloccante — verifica opzionale)** se il
       coder ha tempo: rieseguire il confronto interno
       (`ratio_interior_max`) con `solve_lane_emden(1.5,
       n_points=200000)` (invece di 50000) a parità di `N_SHELLS`, per
       verificare se il residuo mediano interno (~0.001, osservato
       "piatto", non chiaramente decrescente con N nella REV. 2) è
       dominato dalla risoluzione FISSA della griglia di Lane-Emden
       piuttosto che da `N_shells`; se confermato, aggiungere una riga di
       nota nel commento del test. Se non confermato, o se il coder
       preferisce non approfondire, procedere comunque e riportare
       l'osservazione come punto aperto al subagent `critic-fisico` (non
       bloccante per la chiusura del ciclo).
3.6. `test_super_chandrasekhar_triggers_collapse`: caso s20 (catalogo,
     `n=2.0`, `N_SHELLS_TEST_DEFAULT`, `t_max_s =
     T_MAX_FREE_FALL_MULTIPLIER*free_fall_time_s(densita_centrale_gcm3)`);
     `assert result.collapsed is True`; `assert result.t_collapse_s is
     not None and 0 < result.t_collapse_s <= t_max_s`; `assert
     result.rho_c_gcm3_t[-1] > result.rho_c_gcm3_t[0]` (compressione
     reale avvenuta, non solo soglia raggiunta banalmente); **(NUOVO,
     REV. 2)** `assert result.collapse_reason in ("r_min_threshold",
     "shell_crossing")` — il piano non forza a priori quale dei due
     meccanismi scatti per primo in questo caso (dipende dal risultato
     numerico reale), il test verifica solo che il campo sia valorizzato
     coerentemente con `collapsed=True`.
3.7. `test_sub_chandrasekhar_does_not_trigger_collapse`: politropo n=3/2
     autoconsistente con `M_SUB_CHANDRA_TEST_MSUN=0.005` **(REV. 2,
     correzione bloccante 1 — era 0.5 in REV. 1)**, `YE_TEST=0.5`, stessa
     costruzione di 3.5 (incluso `solve_lane_emden(1.5,
     n_points=N_POINTS_LANE_EMDEN_CONVERGENCE)` e il controllo `x_c <
     X_C_SANITY_MAX` — atteso `x_c≈0.0375`, ancora più profondamente
     non-relativistico del caso di 3.5, perché `M_SUB_CHANDRA_TEST_MSUN <
     M_TEST_MSUN`) con `N=N_SHELLS_TEST_DEFAULT`; `t_max_s` calcolato
     dalla densità centrale di QUESTA configurazione con
     `T_MAX_FREE_FALL_MULTIPLIER=10.0`; `assert result.collapsed is
     False`; `assert result.t_collapse_s is None`; `assert
     result.collapse_reason is None`.
     **Nota obbligatoria per il coder (REV. 2, correzione consigliata
     4)**: dopo la correzione del punto 1, il residuo idrostatico
     iniziale di questa configurazione è atteso molto più piccolo
     (~0.1% invece di ~46% del piano REV. 1) — il rischio di "falso
     collasso" per accumulo di residuo numerico su `10*t_ff` è quindi
     molto ridotto, ma va **verificato empiricamente eseguendo il test**,
     non assunto. Se il test fallisse per un falso collasso residuo, il
     coder deve prima verificare (in quest'ordine): (a) che `x_c` sia
     effettivamente sotto `X_C_SANITY_MAX` come atteso; (b) l'andamento
     di `rho_c_gcm3_t` nel tempo (deve restare sostanzialmente piatto,
     non crescere monotonicamente); solo se il problema persiste
     nonostante (a)/(b) confermati, può ridurre
     `T_MAX_FREE_FALL_MULTIPLIER` per questo test specifico (non il
     default globale usato nel demo/3.6), motivando esplicitamente la
     scelta finale nel report al subagent `critic-fisico` (non silenziare
     il problema riducendo la tolleranza o rimuovendo l'assert).
3.8. (aggiuntivo, buona pratica) `test_free_fall_time_scaling`:
     `abs(free_fall_time_s(4*rho)/free_fall_time_s(rho) - 0.5) < 1e-10`
     per un `rho` arbitrario (es. `1e9`) — verifica analitica dello
     scaling `t_ff ∝ rho^{-1/2}`.
3.9. (aggiuntivo, idem) `test_invalid_inputs_raise`: `n_shells<2`,
     `rho_gcm3<=0` in `free_fall_time_s`, `r_min_frac` fuori `(0,1)`,
     `t_max_s<=0`, array di lunghezza mismatched in
     `build_initial_shells`/`simulate_collapse` →
     `pytest.raises(ValueError)` per ciascun caso.
3.10. **(NUOVO, REV. 2, correzione bloccante 2 — copertura diretta del
      nuovo meccanismo)** `test_shell_crossing_event_and_v_floor`:
      - unit test di `dynamics._shell_crossing_event(t, y, n)` (import
        esplicito del privato, stesso precedente di `_gamma1_of_x`): per
        uno stato sintetico con raggi di bordo ben separati e crescenti,
        l'evento deve restituire un valore positivo pari al gap minimo
        atteso (calcolato a mano nel test); per uno stato sintetico con
        due raggi di bordo consecutivi resi artificialmente uguali (shell
        collassata a spessore nullo), l'evento deve restituire
        (circa) zero;
      - unit test di `dynamics._shell_dvdt` con un volume di shell
        forzato a zero (raggi di bordo consecutivi uguali passati
        direttamente alla funzione, bypassando `simulate_collapse`):
        `assert` che la funzione NON sollevi eccezione (comportamento
        atteso dopo la correzione — il floor `V_FLOOR_CM3` la rende
        numericamente sicura) e che il risultato non contenga NaN/inf
        (`np.all(np.isfinite(result))`).

### 4. Chiusura del ciclo
4.1. Eseguire `pytest tests/` (intera suite), riportare solo l'esito
     sintetico (n. test passati/falliti), non il log verboso.
4.2. Eseguire `scripts/step4_demo.py`, verificare che l'output sia
     leggibile, filtrato, e contenga tutti i punti 2.1-2.7, incluso il
     disclaimer obbligatorio.
4.3. Passare la mano al subagent `critic-fisico` con checklist mirata:
     - gravità solo Newtoniana (nessuna correzione relativistica
       introdotta impropriamente in questo step, verificare che
       `_shell_dvdt` non contenga termini GR);
     - nessun termine neutrini, nessuna viscosità artificiale introdotta
       di nascosto;
     - EOS usata nella dinamica è sempre `pressure_chandrasekhar` (mai
       NR/UR come forza motrice);
     - la formula di `polytrope_equilibrium_rho_c_gcm3` corrisponde
       esattamente alla derivazione verificata dal planner (nessuna
       confusione K_NR/K_UR);
     - il check `sol.success`/`RuntimeError` in `simulate_collapse` è
       implementato incondizionatamente;
     - il mismatch noto della condizione iniziale (non equilibrio
       idrostatico esatto rispetto alla EOS reale) è dichiarato
       esplicitamente nel codice/output, non nascosto;
     - nessuna anticipazione di TOV/classificazione remnant (Step 6);
     - **(NUOVO, REV. 2)** nessuna `RuntimeError`/eccezione viene più
       sollevata dentro `_shell_dvdt`/`_collapse_rhs` (il RHS passato a
       `solve_ivp`): verificare che sia stata sostituita dalla coppia
       evento terminale `_shell_crossing_event` + floor `V_FLOOR_CM3`,
       come da correzione bloccante 2;
     - **(NUOVO, REV. 2)** `M_TEST_MSUN=0.01`/`M_SUB_CHANDRA_TEST_MSUN=
       0.005` (non più 1.0/0.5) e `x_c` effettivamente stampato/asserito
       `<X_C_SANITY_MAX` nei test 3.5/3.7, come da correzione bloccante
       1;
     - **(NUOVO, REV. 2)** `solve_lane_emden(1.5, n_points=50000)`
       usato esplicitamente nel test di convergenza 3.5, come da
       correzione 3;
     - **(NUOVO, REV. 2)** `max_step` esplicito passato a `solve_ivp` in
       `simulate_collapse`, come da correzione 4;
     - **(NUOVO, REV. 2)** verificare che il coder abbia effettivamente
       eseguito il test sub-Chandrasekhar (3.7) e riportato l'esito
       empirico su `T_MAX_FREE_FALL_MULTIPLIER=10` nel report, con
       motivazione esplicita se il valore è stato modificato per quel
       test specifico.
     - **(NUOVO, REV. 3)** il test 3.5 esclude un numero FISSO
       (`K_BOUNDARY_EXCLUDE`, k=1 o k=2) di shell di bordo dal residuo di
       convergenza principale, con giustificazione empirica documentata
       nel commento del test (non un numero scelto arbitrariamente);
     - **(NUOVO, REV. 3)** l'assert di monotonicità sul residuo interno
       usa `MONOTONE_TOL_RTOL` (tolleranza stretta, calibrata sul rumore
       reale) al posto della `MONOTONE_SLACK=1.1` superseduta, e
       discrimina effettivamente rumore da trend crescente sistematico
       (verificare leggendo il codice che la tolleranza scelta non sia
       comunque larga al punto da tollerare una crescita sistematica come
       quella osservata in REV. 2);
     - **(NUOVO, REV. 3)** assert separato presente e verificato sulle
       shell di bordo escluse (`ratio_boundary < RATIO_BOUNDARY_SANITY_MAX
       =1.0`), con commento che documenta il valore limite teorico
       `ratio_∞≈0.5656` per n=3/2;
     - **(NUOVO, REV. 3)** `_shell_dvdt`/modulo `dynamics.py` contiene ora
       un docstring esplicito che dichiara il limite strutturale del
       bordo (media di volume + P=0 al bordo libero, non converge a zero
       sull'ultima shell), nello stesso stile delle altre dichiarazioni di
       limite del progetto — verificare che sia effettivamente presente,
       non solo menzionato nel report;
     - **(NUOVO, REV. 3, non bloccante)** se il coder ha eseguito la
       verifica opzionale su `n_points=200000`, verificarne l'esito
       riportato; se non eseguita, prenderne atto come osservazione aperta
       senza bloccare la chiusura del ciclo per questo solo motivo.
4.4. Passare la mano al subagent `reporter` per il resoconto sintetico del
     ciclo.
4.5. Aggiornare questo file: marcare Step 4 come completato in cima,
     aggiungere riga in "Log cicli" con data, esito test, esito
     critic-fisico, valori numerici chiave osservati (es. tempo di
     collasso di s20 vs tempo di caduta libera, densità centrale
     iniziale/finale, `x_c` osservato nei test 3.5/3.7, meccanismo di
     collasso — `collapse_reason` — osservato nel demo e nel test 3.6), e
     per REV. 3: il valore di `K_BOUNDARY_EXCLUDE` scelto e la
     motivazione empirica, il valore di `MONOTONE_TOL_RTOL` scelto, il
     range osservato di `ratio_boundary` rispetto al plateau teorico
     atteso (~0.5656), e l'esito (confermato/non confermato/non eseguito)
     della verifica opzionale su `n_points=200000`.
4.6. **STOP esplicito**: fermarsi qui e attendere conferma esplicita
     dell'utente prima di iniziare Step 5 (stessa regola CLAUDE.md già
     applicata agli step precedenti).

## Log cicli
(ogni ciclo aggiunge una riga qui: data, step completato, note)

### GIORNO 1 — 2026-08-10 — Step 1 completato
**Ambiente**: Python 3.12 installato via winget (fuori dallo scope del coder,
come da piano), virtualenv `.venv` creato in root, dipendenze installate
(numpy, scipy, matplotlib, pandas, pytest).

**Implementato**: modulo `collasso.catalog` con dataclass `Progenitor` e
validazione hard sui parametri fisici (Ye 0.42–0.50, n 1.5–3.0, massa nucleo
1.3–2.0 Msun, coerente con soglia di Chandrasekhar); catalogo di riferimento
`data/progenitors_reference.csv` con 3 progenitori **placeholder** (s15/s20/
s25, fonte bibliografica citata: Sukhbold, Woosley & Heger 2016 — valori
numerici NON estratti dal paper, dichiarati esplicitamente placeholder in
ogni riga, verificato dal test 5.7); script demo `scripts/step1_demo.py`
(eseguito, output verificato leggibile e corretto).

**Test**: `pytest tests/test_catalog.py` → **16/16 passati**, 0 falliti.

**Verifica critic-fisico**: verdetto **"conforme con riserve"**. Tutti i
vincoli applicabili a questo stadio rispettati (nessuna implementazione
prematura di Lane-Emden/EOS/TOV/dinamica, limiti dichiarati onestamente,
range Ye/n/massa validati e testati). Riserva: i 4 parametri del catalogo
placeholder (densità centrale, raggio, n, massa nucleo) non sono
mutuamente consistenti come singolo profilo di Lane-Emden (massa implicita
~5–6x superiore a `massa_nucleo_msun`) — atteso perché presi
indipendentemente da letteratura, ma annotato esplicitamente nel docstring
di `collasso/catalog.py` come avviso vincolante per lo Step 2.

**Pulizia**: passaggio optimizer completato (rimossa una riga ridondante,
aggiunti type hint mancanti; nessun cambio di logica o range fisici).

**Stato progetto**: 1/9 step completati.
**Prossimo step proposto**: Step 2 — Solver equazione di Lane-Emden,
verificato contro le soluzioni analitiche note per n=0,1,5. In attesa di
conferma esplicita dell'utente prima di procedere.

### Nota — 2026-08-10 — Piano Step 2 rivisto dopo review
Il reviewer ha esaminato il piano ciclo corrente (Step 2, REV. 1): verdetto
**positivo** (formule corrette, tolleranze ragionevoli, nessuna violazione
dei vincoli fisici, nessuna riscrittura necessaria). Ha richiesto 4
rifiniture puntuali, ora integrate nel piano (REV. 2, sezione sopra):
1. check esplicito `sol.success`/`RuntimeError` con `sol.message` dopo
   `solve_ivp`, prima di costruire `LaneEmdenSolution` (nuovo punto 1.5);
2. estensione del test `test_solver_properties` a n=3.0 (limite superiore
   del range fisico vincolante del catalogo, punto 3.5);
3. nuovo test di validazione quantitativa `xi1` vs valori tabulati di
   letteratura (Chandrasekhar 1939 / Horedt 2004) per n in
   {1.5, 2.0, 2.5, 3.0}, tolleranza relativa 1e-3 (nuovo punto 3.6);
4. due note di documentazione esplicite: (a) `solution.theta` non ha
   clipping a >=0, residuo numerico atteso ~1e-10 vicino a xi1, da
   documentare nel docstring del campo e verificare in test con
   `theta.min() > -1e-8` (punti 1.1 e 3.5); (b) il test di autoconsistenza
   di `physical_profile` è un controllo algebrico dell'inversione, non una
   validazione fisica indipendente della formula (punto 3.7).
Nessuna riscrittura strutturale: piano pronto per il coder. In attesa di
conferma esplicita dell'utente prima di avviare l'implementazione dello
Step 2.

### GIORNO 2 — 2026-08-10 — Step 2 completato
**Implementato**: `collasso/constants.py` (costanti cgs `G_CGS`, `M_SUN_G`,
`KM_CM`); `collasso/lane_emden.py` — solver dell'equazione di Lane-Emden
(`scipy.integrate.solve_ivp`, RK45, innesco in serie a `xi0=1e-4`, evento
terminale per la superficie, check esplicito `sol.success` → `RuntimeError`
in caso di fallimento), `theta_analytic`/`xi1_analytic` (n=0,1),
`physical_profile` che deriva la scala fisica (α, raggio) usando
`(densita_centrale_gcm3, massa_nucleo_msun)` del catalogo come coppia
primaria (mai `raggio_iniziale_km`, usato solo come cross-check, come
deciso in Step 1). `scripts/step2_demo.py` eseguito: valida n=0,1,5 e
calcola il profilo fisico della stella scelta (s20), con la riga di
disclaimer obbligatoria presente nell'output.

**Test**: `pytest tests/` (intera suite) → **30/30 passati**, 0 falliti
(16 Step 1 + 14 Step 2, inclusi i test contro i valori tabulati di
letteratura Chandrasekhar 1939/Horedt 2004 per ξ1 a n=1.5/2.0/2.5/3.0).
Nessuna regressione su Step 1.

**Verifica critic-fisico**: verdetto **"conforme"** (nessuna riserva).
Checklist verificata punto per punto: gravità solo newtoniana, `n` fisso
per soluzione (nessuna anticipazione dell'EOS a indice variabile di Step
3), nessun termine neutrini/rotazione, condizioni al contorno di
Lane-Emden rispettate, validazione analitica genuina (verificata leggendo
il codice, non solo il report del coder), formula di `physical_profile`
dimensionalmente corretta (verificata anche con un secondo integratore,
Radau — controllo indipendente ad-hoc del critic-fisico durante la
verifica, non un test automatizzato nella suite). Lo scarto del 46% tra
raggio derivato (1026.9 km) e `raggio_iniziale_km` del catalogo (1900 km)
per la stella scelta è stato confermato quantitativamente coerente con
l'incoerenza nota dei 4 parametri del catalogo placeholder già segnalata
in Step 1 (la massa implicita dal raggio catalogo sarebbe ~9.5 Msun contro
1.5 Msun dichiarati) — **è un limite noto dei dati placeholder, non un
difetto del solver**, e non verrà "risolto" automaticamente dallo Step 3:
resterà tale finché non saranno disponibili i file del catalogo completo.

**Pulizia**: passaggio optimizer completato — fix cosmetico dell'output
demo (rimosso uno scarto relativo fuorviante vicino agli zeri di θ per
n=0,1) e piccola pulizia di stile; nessun cambio di logica, formule
fisiche o tolleranze. Suite ri-verificata dopo la pulizia: ancora 30/30.

**Stato progetto**: 2/9 step completati.
**Prossimo step proposto**: Step 3 — EOS di Chandrasekhar (non-rel. ->
ultra-rel.). In attesa di conferma esplicita dell'utente prima di
procedere.

### Nota — 2026-08-10 — Piano Step 3 pianificato
Il planner ha scomposto lo Step 3 (EOS di Chandrasekhar) in task granulari
(sezione "Piano ciclo corrente (Step 3)" sopra), fissando esplicitamente
tutte le tolleranze numeriche di validazione non lasciate a discrezione del
coder: `TOL_ASYMPT_RTOL=1e-6` per gli asintoti pure-x di `chandrasekhar_f`
e `_gamma1_of_x` a x=1e-4/1e4 (con verifica analitica degli ordini di
grandezza attesi, ~1e-8, riportata nel piano); range `[1.3, 1.6]` Msun per
`chandrasekhar_mass_msun(0.5)` (valore atteso ≈1.458 Msun, cross-check con
la relazione di letteratura M_Ch≈5.83·Ye²); `H_FD_REL=1e-4` e
`TOL_GAMMA1_FD_RTOL=1e-5` per il cross-check a differenze finite di Gamma1
a rho={1e6,1e7} g/cm^3 (regione di transizione, non asintotica). Fissata
anche la formula chiusa di inversione x->rho per la costruzione delle
griglie di demo/test, e la necessità di un helper privato `_gamma1_of_x`
(non nella lista pubblica concordata) per poter testare gli asintoti di
Gamma1 "in forma pura su x" come richiesto. In attesa di conferma esplicita
dell'utente (ed eventuale passaggio al reviewer) prima di avviare
l'implementazione dello Step 3.

### Nota — 2026-08-10 — Piano Step 3 rivisto dopo review (fix cancellazione catastrofica)
Il reviewer ha verificato analiticamente tutte le formule fisiche e le
costanti CODATA del piano Step 3 (REV. 1): **tutte corrette, nessun errore
fisico**. Ha però individuato un problema numerico serio e concreto: la
valutazione diretta di `chandrasekhar_f(x)` per `x` piccolo (es. il punto
di test degli asintoti `x=1e-4`) soffre di **cancellazione catastrofica**
(due termini di segno opposto ~∓3e-4 il cui risultato atteso è ~1.6e-20),
con errore relativo atteso ben oltre la tolleranza `TOL_ASYMPT_RTOL=1e-6`
già fissata — non un problema di formula fisica, ma di implementazione in
floating point. Il piano è stato corretto (REV. 2, sezione sopra) imponendo:
1. implementazione a due rami per `chandrasekhar_f`, soglia
   `X_SERIES_THRESHOLD=0.1` fissata dal planner (punto 1.2 aggiornato);
2. formula della serie troncata a 3 termini `f_series(x) = (8/5)*x**5 -
   (4/7)*x**7 + (1/3)*x**9`, derivata e verificata dal planner, fissata
   esattamente così (non a discrezione del coder);
3. propagazione automatica della correzione a `_gamma1_of_x`/
   `gamma1_chandrasekhar` senza ramo separato (punto 1.7, nota aggiunta);
4. il punto di test `x=1e-4` e la tolleranza `TOL_ASYMPT_RTOL=1e-6` restano
   invariati (ora affidabili grazie al fix), nessuno spostamento necessario;
5. nuovo test dedicato `test_chandrasekhar_f_series_branch_continuity`
   (punto 3.10) che confronta i due rami appena sotto/sopra la soglia
   (x=0.099/0.101) contro formule di riferimento indipendenti scritte nel
   test stesso, con nuova tolleranza `TOL_BRANCH_CONTINUITY_RTOL=1e-6`;
6. nota esplicita sulla motivazione del branching (cancellazione
   catastrofica, non overflow/underflow/asinh) da riportare nel docstring
   del codice, già inclusa nel piano.
Nessun'altra parte del piano toccata: formule fisiche, costanti CODATA,
tolleranze di `M_Ch` e del cross-check a differenze finite di `Gamma1`,
perimetro dello step restano quelli già validati. Piano pronto per un
nuovo giro di review/conferma dell'utente prima di avviare l'implementazione
dello Step 3.

### GIORNO 3 — 2026-08-10 — Step 3 completato
**Implementato**: `collasso/eos.py` — EOS esatta di Chandrasekhar per gas di
elettroni completamente degenere (T=0): `chandrasekhar_f(x)` (a due rami,
serie troncata per x<0.1 / formula diretta sopra, per evitare la
cancellazione catastrofica individuata **in fase di review del piano,
prima che il coder scrivesse codice** — non un bug corretto a posteriori
su codice già scritto), `pressure_chandrasekhar`, i limiti
`pressure_non_relativistic` (∝ρ^5/3) e `pressure_ultrarelativistic`
(∝ρ^4/3), `gamma1_chandrasekhar`/`n_eff_chandrasekhar` (transizione
1.5→3.0 verificata nel demo), `chandrasekhar_mass_msun` (riuso genuino di
`solve_lane_emden(3.0)` di Step 2, nessuna costante ξ1 hardcoded —
verificato dal critic-fisico leggendo il codice). Estesa
`collasso/constants.py` con costanti CODATA 2018 (`HBAR_CGS`,
`C_LIGHT_CGS`, `M_ELECTRON_G`, `M_U_G`). Script demo eseguito e verificato:
`chandrasekhar_mass_msun` per i 3 Ye del catalogo — s15 (Ye=0.50) →
**1.4559 Msun** (scarto 3.99% da "~1.4 Msun" di CLAUDE.md), s20 (Ye=0.46)
→ **1.2323 Msun** (scarto 11.98%), s25 (Ye=0.43) → **1.0768 Msun** (scarto
23.09%) — dichiarato esplicitamente valore Newtoniano classico (n=3), non
la soglia GR-corretta (quella arriverà con TOV, Step 6); disclaimer
verificato presente nell'output reale.

**Test**: `pytest tests/` (intera suite) → **41/41 passati**, 0 falliti
(16 Step 1 + 14 Step 2 + 11 Step 3, inclusi gli asintoti su x puro, la
monotonia di P(ρ), il range di M_Ch, il cross-check a differenze finite di
Gamma1, e il test dedicato di continuità del branching numerico a
x=0.099/0.101). Nessuna regressione su Step 1/2.

**Verifica critic-fisico**: verdetto **"conforme"** (nessuna riserva).
Checklist verificata leggendo il codice (non solo il report del coder):
branching a serie corretto e verificato dimensionalmente; formule di
pressione/Gamma1 corrispondenti esattamente al piano; riuso genuino di
`solve_lane_emden(3.0)` confermato; controllo quantitativo indipendente
che M_Ch scala esattamente come Ye² (rapporti 0.8464 e 0.7396, uguali a
(Ye/0.50)² entro precisione macchina — coerente con la relazione di
letteratura M_Ch≈5.83·Ye² Msun, Shapiro & Teukolsky 1983); perimetro dello
step rispettato (nessuna dinamica, nessuna correzione relativistica oltre
la EOS, nessun TOV); disclaimer onesti presenti nel codice e nell'output.

**Pulizia**: passaggio optimizer completato (helper per pattern ripetuti,
costante `_3_PI2` per evitare ricalcoli, docstring migliorate); nessun
cambio di logica, formule fisiche, costanti o tolleranze. Suite
ri-verificata dopo la pulizia: ancora 41/41.

**Nota di processo**: il subagent `reporter` aveva omesso, per 3 cicli
consecutivi (Step 1, 2, 3), l'esito test e il verdetto critic-fisico dal
resoconto — corretto qui manualmente. Il template del subagent
(`.claude/agents/reporter.md`) è stato aggiornato aggiungendo campi
"Test" e "Critic-fisico" obbligatori, per prevenire la recidiva nei
prossimi cicli.

**Stato progetto**: 3/9 step completati.
**Prossimo step proposto**: Step 4 — Dinamica shell Lagrangiane, loop
temporale del collasso. Nota: la EOS P(ρ) reale e numericamente stabile
di questo step è ora disponibile per sostituire l'approssimazione a n
fisso di Lane-Emden nella dinamica. In attesa di conferma esplicita
dell'utente prima di procedere.

### Nota — 2026-08-10 — Piano Step 4 pianificato
Il planner ha scomposto lo Step 4 (dinamica shell Lagrangiane) in task
granulari (sezione "Piano ciclo corrente (Step 4)" sopra), a partire
dall'approccio numerico già approvato dall'utente (schema von
Neumann–Richtmyer, gravità Newtoniana + gradiente di pressione da
`pressure_chandrasekhar`, evento terminale a `r_min_frac`). Il planner ha
verificato algebricamente in modo indipendente la formula chiusa
`polytrope_equilibrium_rho_c_gcm3` (derivazione da
`d/dr(r²/ρ·dP/dr)=-4πG·r²·ρ` con sostituzione di Lane-Emden, per n=3/2),
confermando che coincide esattamente con quella fornita nell'approccio
approvato — nessun errore trovato. Ha inoltre identificato e documentato
esplicitamente il rischio di confusione fra `K_NR` (limite non-relativistico,
necessario per questa formula) e `K_UR`/`k_ultrarelativistic` di
`collasso/eos.py` (limite ultra-relativistico, n=3, usato altrove per
`chandrasekhar_mass_msun`), fissando il metodo esatto per ottenere `K_NR`
senza duplicare formule: `eos.pressure_non_relativistic(1.0, ye)`.
Fissati numericamente tutti i parametri lasciati aperti dall'approccio
approvato: `R_MIN_FRAC_DEFAULT=0.1`, `T_MAX_FREE_FALL_MULTIPLIER=10.0`
(con giustificazione fisica del multiplo), tolleranze/integratore di
`solve_ivp` (`RK45`, `rtol=1e-6`, `atol=1e-2`), `N_SHELLS_DEMO_DEFAULT=200`,
`N_SHELLS_TEST_DEFAULT=50`, `N_CONVERGENCE_LIST=[50,100,200,400]` con
tolleranze di convergenza (`RATIO_TOL_N400=1e-2`,
`RATIO_REDUCTION_FACTOR_MIN=4.0`) coerenti con uno scaling dell'errore di
discretizzazione atteso fra O(1/N) e O(1/N²), massa/Ye del caso test di
equilibrio autoconsistente (`M_TEST_MSUN=1.0`, `YE_TEST=0.5`) e del caso
sub-Chandrasekhar ad hoc (`M_SUB_CHANDRA_TEST_MSUN=0.5`), e l'algoritmo
esatto (non ambiguo) di `build_initial_shells` (integrazione trapezoidale
con `scipy.integrate.cumulative_trapezoid`, shell equispaziate in massa,
interpolazione inversa con `np.interp`). In attesa di conferma esplicita
dell'utente (ed eventuale passaggio al reviewer) prima di avviare
l'implementazione dello Step 4.

### Nota — 2026-08-11 — Piano Step 4 rivisto dopo review (REV. 2)
Il reviewer ha esaminato il piano ciclo corrente (Step 4, REV. 1) e ha
trovato 4 problemi concreti (2 bloccanti, 2 da chiarire/consigliati), ora
corretti nel piano sopra (sezione "Piano ciclo corrente (Step 4) — REV.
2"):
1. **(bloccante)** le masse di test `M_TEST_MSUN=1.0`/
   `M_SUB_CHANDRA_TEST_MSUN=0.5` della REV. 1 erano in realtà nella
   regione di transizione relativistica (`x_c≈1.28` per `M_TEST_MSUN=
   1.0`, non "profondamente non-relativistico" come assunto), con uno
   scarto fisico (~46%) fra `pressure_chandrasekhar` e
   `pressure_non_relativistic` che avrebbe invalidato sia il test di
   convergenza (3.5) sia il test sub-Chandrasekhar (3.7). Corretto
   riducendo a `M_TEST_MSUN=0.01`, `M_SUB_CHANDRA_TEST_MSUN=0.005`
   (`x_c≈0.06`/`≈0.0375`), con verifica esplicita di `x_c <
   X_C_SANITY_MAX=0.1` richiesta al coder nei test stessi (nuova
   costante 3.1, punti 3.5/3.7 aggiornati).
2. **(bloccante)** `RuntimeError` sollevata dentro `_shell_dvdt` (il RHS
   passato a `solve_ivp`) non viene catturata da scipy — crash non
   gestito. Corretto sostituendo con un secondo evento terminale
   `_shell_crossing_event` (stesso pattern di `_collapse_threshold_event`)
   più un floor numerico `V_FLOOR_CM3=1e-30` dentro il RHS (mai
   un'eccezione); aggiunto il campo `collapse_reason` a
   `CollapseSolution` per distinguere `"r_min_threshold"` da
   `"shell_crossing"`; nuovo test dedicato 3.10.
3. **(da chiarire)** risoluzione di griglia di Lane-Emden insufficiente a
   `N=400` shell col default `n_points=2000` (~5 punti/shell). Corretto
   imponendo esplicitamente `solve_lane_emden(1.5,
   n_points=N_POINTS_LANE_EMDEN_CONVERGENCE=50000)` (~125 punti/shell a
   N=400) nel test di convergenza 3.5.
4. **(consigliato)** nessun safeguard sul costo computazionale/step-size
   di `solve_ivp`. Corretto aggiungendo `max_step=t_max_s/2000` esplicito
   in `simulate_collapse`, più una nota obbligatoria per il coder di
   verifica empirica (non solo assunta) che `T_MAX_FREE_FALL_MULTIPLIER=
   10` non produca un falso collasso nel caso sub-Chandrasekhar corretto
   (punto 3.7), con eventuale aggiustamento motivato nel report al
   subagent critic-fisico.
Nessun'altra parte del piano è stata toccata: schema di discretizzazione
(shell Lagrangiane, gravità Newtoniana + pressione da
`pressure_chandrasekhar`), formula di `polytrope_equilibrium_rho_c_gcm3`
(verificata algebricamente dal planner, invariata) e perimetro fisico
dello step restano quelli già validati dal reviewer. Piano pronto per un
nuovo giro di review/conferma dell'utente prima di avviare l'implementazione
dello Step 4.

### Nota — 2026-08-11 — Piano Step 4 rivisto dopo review (REV. 3, correzione del test 3.5)
Il coder ha implementato la REV. 2 alla lettera: 49/50 test passano,
`test_polytrope_equilibrium_initial_acceleration_convergence` (3.5)
fallisce **genuinamente**, non per errore di trascrizione. Il reviewer ha
diagnosticato e quantificato analiticamente un limite strutturale reale
dello schema di discretizzazione al bordo libero (media di volume
sull'ultima shell + P=0 al bordo, per un profilo a densità nulla in
superficie il residuo converge a un plateau universale non nullo,
`|1-2/(n+1)^((n+1)/n)|`, ≈0.5656 per n=3/2 — non risolvibile aumentando
N), confermato dal planner leggendo direttamente `collasso/dynamics.py`
(`_shell_dvdt`, righe ~231-261, dove `rho[-1] = delta_m_g[-1]/v[-1]` è
verificata essere effettivamente una densità media di volume). Il
reviewer ha anche identificato un problema logico separato nella metrica
del test attuale: la tolleranza `MONOTONE_SLACK=1.1` tollera "per
sbaglio" una sequenza sistematicamente crescente verso quel plateau,
senza discriminarla da una convergenza rumorosa genuina.

Il piano è stato corretto (REV. 3, sezione sopra) SOLO nella metrica del
test 3.5 e nella documentazione — **nessuna modifica alla fisica o alla
discretizzazione `_shell_dvdt`/`_collapse_rhs`**, già validate e non
rimesse in discussione:
1. il residuo di convergenza principale del test 3.5 ora esclude un
   numero fisso e piccolo di shell di bordo (`K_BOUNDARY_EXCLUDE`, k=1 o
   k=2 — il coder deve determinare empiricamente il valore minimo
   sufficiente, provando entrambi su `N_CONVERGENCE_LIST`, e documentarlo
   nel codice);
2. nuovo assert separato e più debole (`RATIO_BOUNDARY_SANITY_MAX=1.0`)
   sulle shell di bordo escluse, per documentare il limite come fatto
   testato esplicitamente (non solo descritto in prosa);
3. l'assert di monotonicità è stato corretto con una tolleranza stretta
   dedicata (`MONOTONE_TOL_RTOL`, da calibrare empiricamente dal coder in
   una fascia indicativa 0.5%-2% per passo), in sostituzione della
   `MONOTONE_SLACK=1.1` superseduta che non discriminava correttamente;
4. nuovo requisito di docstring esplicito in `dynamics.py` (punto 1.4bis)
   che dichiara il limite strutturale del bordo, nello stesso stile delle
   dichiarazioni già presenti per neutrini/viscosità/rotazione;
5. verifica opzionale non bloccante segnalata al coder: rieseguire con
   `n_points=200000` (invece di 50000) a `N_shells` fisso, per capire se
   il residuo mediano interno "piatto" (~0.001) osservato nella REV. 2 è
   dominato dalla risoluzione fissa della griglia di Lane-Emden piuttosto
   che da `N_shells`; da riportare come osservazione al critic-fisico in
   ogni caso (confermata o meno).
Piano pronto per l'implementazione della correzione da parte del coder,
seguita da nuova verifica del critic-fisico prima della chiusura del
ciclo Step 4.

### Nota — 2026-08-11 — Piano Step 4 rivisto dopo indagine forense (REV. 4)
Scritta direttamente dall'orchestratore (non dal subagent planner, per
ridurre l'esposizione a interruzioni di sessione ripetute durante
un'indagine già conclusa) — contenuto comunque derivato dalle verifiche
indipendenti già fatte da reviewer e critic-fisico nei passaggi precedenti,
non da una nuova decisione non verificata.

**Scoperta 1 (confermata quantitativamente, non un bug)**: il residuo
"piatto" nelle shell profondamente interne (10%-50% della massa) del test
3.5 è un vero mismatch FISICO fra l'equilibrio di test (costruito con
`pressure_non_relativistic`, leading order) e la dinamica (che usa
correttamente `pressure_chandrasekhar`, l'EOS esatta, come richiesto dal
progetto — MAI il viceversa). Lo scarto locale segue `-(5/14)*x(r)^2`
(Taylor di Step 3), verificato in accordo a 3-4 cifre significative col
residuo dinamico osservato a più frazioni di massa. Controllo decisivo:
sostituendo `pressure_chandrasekhar` con la stessa `pressure_non_relativistic`
usata per l'equilibrio, il residuo crolla di 1-2 ordini di grandezza e
converge correttamente con N (non più un plateau) — prova diretta che non
è un limite dello schema, ma del caso di test (x_c=0.0593 non abbastanza
piccolo).

**Scoperta 2 (confermata quantitativamente, un secondo limite strutturale
distinto, non un bug)**: la shell più interna (indice 0, il centro) mostra
un residuo (~5.4%-5.6%) non spiegato dalla Scoperta 1 (identico anche con
EOS "abbinata" NR/NR) e non dovuto a cancellazione/precisione numerica
(valori assoluti lontani da underflow; convergenza liscia e monotona su 4
ordini di grandezza di N, 50→6400: 0.05434, 0.05519, 0.05536, 0.05560).
Converge verso un plateau finito non nullo (~0.056-0.058) — stessa natura
qualitativa del limite già noto e dichiarato al bordo libero esterno
(~0.5656), ma un fenomeno distinto, al bordo INTERNO (centro), mai
diagnosticato prima. Confermato dall'utente sui numeri grezzi.

**Correzione REV. 4 (approvata esplicitamente dall'utente)**:
1. Ridurre `M_TEST_MSUN` da `0.01` a **`0.001`** (fattore 10) e
   `M_SUB_CHANDRA_TEST_MSUN` da `0.005` a **`0.0005`** (stesso rapporto
   0.5), per portare `x_c` da ≈0.0593 a **≈0.0128** (scaling verificato
   empiricamente `x_c ∝ M^(2/3)`), riducendo il mismatch fisico residuo
   `-(5/14)*x_c^2` da ≈0.126% a **≈0.006%** — un ordine di grandezza sotto
   la nuova tolleranza (vedi punto 3). Verificare `x_c < X_C_SANITY_MAX=0.1`
   come già richiesto (invariato).
2. Sostituire la metrica di convergenza del test 3.5 con una metrica
   AGGREGATA pesata in massa, calcolata su TUTTE le shell (nessuna
   esclusione manuale per indice — `K_BOUNDARY_EXCLUDE`,
   `RATIO_BOUNDARY_SANITY_MAX`, `MONOTONE_TOL_RTOL`, `ratio_interior`/
   `ratio_boundary` della REV. 3 sono SUPERSEDUTI, mantenerli solo come
   nota storica nel commento, non nella logica del test):
   ```
   global_ratio = sqrt(sum(delta_m_i * a_net_i**2)) / sqrt(sum(delta_m_i * a_grav_i**2))
   ```
   Per costruzione sopprime il contributo delle shell di bordo (centro E
   superficie), che pesano O(1/N) in massa ciascuna, mentre il bulk
   interno (che converge correttamente, Scoperta 1) domina la somma.
3. Tolleranze per `global_ratio` (nuove costanti, sostituiscono
   `RATIO_TOL_N400`/`RATIO_REDUCTION_FACTOR_MIN` per questo test):
   `GLOBAL_RATIO_TOL_N400 = 1e-3` (un ordine di grandezza sopra il nuovo
   floor fisico atteso ≈6e-5, margine di sicurezza ≈17x) e
   `GLOBAL_RATIO_REDUCTION_FACTOR_MIN = 3.0` fra N=50 e N=400 (più
   permissivo del precedente 4.0, perché la metrica aggregata include
   comunque un contributo residuo dal bordo esterno che non scala come
   pura convergenza O(1/N²) del bulk, ma tipicamente più lentamente).
   **Il coder deve calcolare empiricamente `global_ratio` per
   N=[50,100,200,400] PRIMA di fissare in modo definitivo questi due
   numeri nel test**: se i valori osservati non rispettano comodamente
   queste soglie con margine ragionevole, il coder è autorizzato a
   proporre valori diversi ma deve riportare esplicitamente i numeri
   osservati (non silenziare un adattamento delle tolleranze senza
   motivazione tracciata) — l'obiettivo è chiudere il ciclo con numeri
   veri, non indovinati a tavolino una seconda volta.
4. Documentare ENTRAMBI i limiti strutturali (bordo esterno libero E bordo
   interno/centro) nel docstring di `collasso/dynamics.py` (modulo e/o
   `_shell_dvdt`) — il centro va aggiunto ora, non era documentato dalla
   REV. 3 (che discuteva solo il bordo esterno).
5. Aggiornare anche `M_SUB_CHANDRA_TEST_MSUN` nel test 3.7 al nuovo valore
   `0.0005`; il coder deve rieseguire e confermare (non assumere) che
   `T_MAX_FREE_FALL_MULTIPLIER=10` resti valido (nessun falso collasso) con
   la massa più piccola.
6. Nessuna modifica alla fisica/discretizzazione (`_shell_dvdt`,
   `_collapse_rhs`, formula di `polytrope_equilibrium_rho_c_gcm3`) né al
   test 3.6 (super-Chandrasekhar, già passa) — tutti invariati e non
   rimessi in discussione.

Piano pronto per l'implementazione da parte del coder, seguita da
verifica del critic-fisico prima della chiusura del ciclo Step 4.

### GIORNO 4 — 2026-08-11 — Step 4 completato
**Nota di processo**: a causa di ripetute interruzioni di sessione durante
questo ciclo (limite di utilizzo raggiunto più volte), gli ultimi passaggi
di chiusura (formalizzazione REV. 4, verifica finale, questo log) sono
stati eseguiti direttamente dall'orchestratore invece che dai subagent
`optimizer`/`reporter`, per ridurre il rischio di ulteriori interruzioni
su lavoro a basso rischio/informativo. Il contenuto tecnico (diagnosi,
correzioni, verifica critic-fisico) è comunque passato per il ciclo
normale di verifica indipendente (reviewer + critic-fisico, con controlli
numerici rieseguiti in autonomia da entrambi).

**Implementato**: `collasso/dynamics.py` — dinamica shell Lagrangiane a
griglia sfalsata (stile von Neumann-Richtmyer), gravità Newtoniana +
gradiente di pressione dall'EOS esatta di Chandrasekhar (Step 3, MAI i
limiti NR/UR come motore della dinamica), integrazione con
`scipy.solve_ivp` (due eventi terminali: soglia centrale `r_min_frac`,
shell crossing; floor numerico `V_FLOOR_CM3` per sicurezza, mai
un'eccezione nel RHS). `scripts/step4_demo.py`: per la stella scelta
(s20, Ye=0.46), `massa_nucleo_msun=1.50` confrontata con
`chandrasekhar_mass_msun(0.46)=1.2323` Msun (supra-Chandrasekhar, scarto
+21.7%) — collasso innescato e verificato (`collapsed=True`,
`t_collapse≈99.5 ms ≈ 4.1×t_ff`, `collapse_reason='r_min_threshold'`),
densità centrale cresciuta di ~3 ordini di grandezza (7.34e9 → 6.49e12
g/cm³).

**Test**: `pytest tests/` (intera suite) → **50/50 passati**, 0 falliti
(16+14+11+9 dai primi tre step + 9 nuovi di Step 4, incluso il test di
equilibrio autoconsistente n=3/2 riprogettato in REV. 4 con metrica
aggregata pesata in massa, il test super-Chandrasekhar — collasso atteso
e verificato — e il test sub-Chandrasekhar — nessun falso collasso,
rieseguito con la massa corretta). Nessuna regressione su Step 1-3.

**Percorso del ciclo (insolitamente lungo, per trasparenza)**: il piano
REV. 2 (schema numerico, formule, tolleranze) è stato validato
correttamente dal reviewer fin da subito. L'implementazione ha però
rivelato **due limiti strutturali genuini** dello schema di
discretizzazione a griglia sfalsata, entrambi diagnosticati
quantitativamente (non bug, non risolvibili aumentando N):
1. **Bordo libero esterno** (superficie, densità nulla): il rapporto
   |accelerazione netta|/gravità sull'ultima shell converge a un plateau
   universale `|1-2/(n+1)^((n+1)/n)|≈0.5656` (per n=3/2), non a zero —
   causato dall'uso di densità media di volume invece che puntuale al
   bordo, combinato con la caduta a legge di potenza della densità in
   superficie.
2. **Bordo interno/centro**: un effetto analogo, distinto, mai
   diagnosticato prima, converge a un plateau ~0.056-0.058 alla shell più
   interna.
Un'indagine forense approfondita (reviewer + critic-fisico, con script
diagnostici indipendenti, verificati anche dall'utente sui numeri grezzi)
ha inoltre isolato e quantificato un mismatch fisico separato (non un
bug): il caso di equilibrio di test, costruito con l'EOS non-relativistica
pura, differisce leggermente dalla dinamica (che usa correttamente l'EOS
esatta) — scarto locale `-(5/14)x(r)²`, confermato analiticamente e
numericamente. Risolto (REV. 4) riducendo `M_TEST_MSUN` a 0.001 Msun
(x_c≈0.0128, mismatch trascurabile) e sostituendo la metrica di
convergenza per-shell con una metrica aggregata pesata in massa, le cui
tolleranze sono state calibrate empiricamente sui valori osservati (non
indovinate a tavolino), con piena trasparenza sui numeri nel codice.

**Verifica critic-fisico**: verdetto finale **"conforme con riserve"**.
Perimetro fisico rispettato integralmente e verificato riga per riga
(gravità solo Newtoniana, EOS esatta come unico motore della dinamica,
nessun neutrini/relativistico/rotazione/viscosità introdotto). I due
limiti strutturali sono documentati onestamente nel codice (docstring di
modulo e di `_shell_dvdt`), non camuffati da bug. La riserva è puramente
*diagnostica*, non fisica: la metrica aggregata del test di equilibrio è
risultata empiricamente dominata (>99%) dal plateau di bordo esterno già
noto, non dal bulk interno come previsto dal piano — il test certifica
quindi "il plateau di bordo si comporta come atteso", non più "l'intero
schema converge all'equilibrio". Segnalato come possibile lavoro futuro
(un test più mirato e isolato sul bulk), non bloccante per questo ciclo.

**Pulizia**: passaggio optimizer separato saltato in questo ciclo (nota
di processo sopra); il codice del coder è già stato scritto con
attenzione a docstring/naming durante le correzioni REV. 2-4.

**Stato progetto**: 4/9 step completati.
**Prossimo step proposto**: Step 5 — Correzioni relativistiche sul nucleo.
In attesa di conferma esplicita dell'utente prima di procedere.

## Piano ciclo corrente (Step 5)
Scritto direttamente dall'orchestratore (non dal subagent planner, per
ridurre l'esposizione a interruzioni durante chiamate lunghe — richiesta
esplicita dell'utente di uno svolgimento lineare). Contenuto derivato dal
piano già approvato dall'utente in plan mode, con tutte le costanti
numeriche fissate qui, non lasciate a discrezione del coder.

**Scelta di design (perché questo step resta lineare)**: a differenza di
Step 4, non si introduce un nuovo schema di discretizzazione. Si modifica
solo il termine sorgente di gravità già esistente in `_shell_dvdt`
(Step 4), sostituendo `-G*M/r²` con una correzione relativistica
algebrica locale per shell (formula chiusa, valutata punto per punto).
Non c'è quindi un nuovo studio di convergenza in N da aprire.

**Riferimento fisico**: potenziale gravitazionale effettivo "Case A" di
Marek, Dimmelmeier, Janka, Müller & Buras (2006, A&A 445, 273) — standard
di riferimento per correzioni relativistiche approssimate in codici
Newtoniani di collasso stellare, derivato dalla riduzione dell'equazione
TOV al limite post-Newtoniano:
```
a_grav_GR(r) = -G * [m(r) + 4*pi*r^3*P(r)/c^2] * [1 + P(r)/(rho(r)*c^2)]
               / (r^2 * [1 - 2*G*m(r)/(r*c^2)])
```
Riduce esattamente a `-G*m/r²` (Newtoniano, Step 4) quando
`P/(rho*c²)->0` e `2*G*m/(r*c²)->0` (campo debole).

### 1. Modulo `collasso/relativistic.py` (nuovo)
1.1. `compactness(m_g, r_cm) -> float`: `2*G_CGS*m_g/(r_cm*C_LIGHT_CGS**2)`.
     Guardia: `ValueError` se `r_cm<=0` o `m_g<=0` (stesso stile delle
     guardie già usate in `collasso.dynamics`/`collasso.eos`).
1.2. `relativistic_grav_accel(m_g, r_cm, p_cgs, rho_gcm3) -> float`: la
     formula sopra esattamente come scritta.
     **CORREZIONE (da review): MAI sollevare un'eccezione qui.** Questa
     funzione è chiamata da `_shell_dvdt`, che è sul path del RHS passato
     a `solve_ivp` — esattamente come già scoperto e corretto in Step 4
     REV. 2 (`_shell_dvdt` V_FLOOR_CM3), un'eccezione dentro il RHS NON
     viene catturata da scipy e causa un crash non gestito, perché gli
     eventi di `solve_ivp` sono controllati fra step accettati, non ad
     ogni valutazione interna dello stadio RK45 (il solver può valutare il
     RHS in stati di prova oltre la soglia prima che l'evento scatti).
     Usa invece un CLAMP numerico: `compactness_clamped =
     min(compactness(m_g, r_cm), COMPACTNESS_CLAMP_MAX)` (nuova costante,
     vedi 3.1) prima di valutare `[1 - 2*G*m/(r*c^2)]` nel denominatore —
     stesso principio di `V_FLOOR_CM3`, mai un'eccezione, solo un limite
     numerico di sicurezza residuo (l'evento terminale al punto 2.2 resta
     l'unico meccanismo con cui l'esito viene segnalato all'esterno).
1.3. `rho_gcm3` è la densità della shell già disponibile nel chiamante
     (`_shell_dvdt`) — non ricalcolarla qui, passarla come parametro.

### 2. Estensione di `collasso/dynamics.py` (non riscrittura)
2.1. `_shell_dvdt(r_cm, m_enclosed_g, delta_m_g, ye, relativistic: bool =
     False)`: se `relativistic=False` (default), comportamento
     IDENTICO a Step 4 (nessuna riga di logica gravitazionale cambiata sul
     path di default — il branch `if relativistic:` deve essere una
     semplice sostituzione del termine `grav`, non un refactor della
     funzione). Se `True`, sostituisce
     `grav = -G_CGS*m_enclosed_g/r_cm**2` con
     `grav[i] = -relativistic_grav_accel(m_enclosed_g[i], r_cm[i], p[i], rho[i])`
     per ogni shell i (vettorizzato, riusando gli array `p`/`rho` già
     calcolati nella funzione per il termine di pressione — NESSUNA
     modifica al termine di pressione `press_int`/`press_ext`, che resta
     invariato, limite dichiarato esplicitamente nel docstring: la
     correzione si applica solo al termine sorgente gravitazionale, non è
     un solve GR completo dell'idrodinamica).
2.2. `simulate_collapse(..., relativistic: bool = False)`: propaga il
     flag a `_shell_dvdt` tramite gli `args` di `solve_ivp` — **nota
     esplicita (da review)**: `solve_ivp` passa un'UNICA tupla `args`
     comune a `fun` e a TUTTI gli eventi (vincolo già scoperto e
     documentato in Step 4, vedi commento in `_collapse_rhs`); allargare
     questa tupla comune per includere `relativistic` tocca quindi le
     firme di `_collapse_rhs`, `_collapse_threshold_event`,
     `_shell_crossing_event` E del nuovo `_schwarzschild_proximity_event`
     — stesso pattern già stabilito, non da riscoprire.
     Se `relativistic=True`, aggiunge un TERZO evento terminale
     `_schwarzschild_proximity_event`: **CORREZIONE (da review)** non
     controlla solo la shell 0, ma calcola la compattezza di TUTTE le
     shell (vettorizzato) e ritorna `COMPACTNESS_SAFETY_LIMIT -
     np.max(compactness_di_tutte_le_shell)` (`direction=-1`,
     `terminal=True`) — garantisce che l'evento scatti su QUALUNQUE shell
     si avvicini per prima alla soglia, non solo la più interna (il
     piano REV. 1 assumeva senza dimostrazione che la shell 0 avesse
     sempre la compattezza massima). Nuovo `collapse_reason=
     "near_schwarzschild"` in `CollapseSolution` (aggiungere questo valore
     alla lista di valori ammessi nel docstring della dataclass). Se
     `relativistic=False`, il terzo evento non viene nemmeno costruito
     (lista eventi identica a Step 4: solo i primi due).

### 3. Costanti (fissate qui, non a discrezione del coder)
3.1. `COMPACTNESS_SAFETY_LIMIT = 0.9` (evento terminale, in
     `collasso/relativistic.py`) — margine di sicurezza prima della
     singolarità formale a compattezza=1, oltre il quale l'approssimazione
     "Case A" perde comunque di significatività fisica. **NUOVA COSTANTE
     (da review)**: `COMPACTNESS_CLAMP_MAX = 0.99` — clamp numerico
     interno usato SOLO dentro `relativistic_grav_accel` come rete di
     sicurezza residua (stesso ruolo di `V_FLOOR_CM3` in Step 4), mai
     un'eccezione; in pratica non dovrebbe mai essere esercitato, perché
     l'evento terminale a 0.9 ferma l'integrazione prima.
3.2. **CORREZIONE (da review): valore incoerente con l'esempio dato nella
     REV. 1 di questo piano.** La deviazione `a_GR/a_Newt - 1` è LINEARE
     in compattezza e in `P/(rho*c²)` (verificato dal reviewer via
     espansione al primo ordine: `≈ 4*pi*r^3*P/(m*c^2) + P/(rho*c^2) +
     compattezza`, nessuna cancellazione che porti a ordine quadratico) —
     quindi a compattezza~1e-6 lo scarto atteso è ~1e-6, non ~1e-8.
     Fissato correttamente: punto di prova a compattezza E `P/(rho*c²)`
     entrambi **~1e-8** (scegliere `r_cm`, `m_g`, `p_cgs`, `rho_gcm3`
     coerentemente per ottenere questo ordine di grandezza su entrambi i
     parametri, non solo sulla compattezza), con
     `TOL_WEAK_FIELD_RTOL = 1e-5` (margine ampio, ~100x-1000x sopra lo
     scarto atteso ~1e-8/2e-8, per assorbire eventuali differenze di
     coefficiente fra i tre termini additivi senza dover calcolare a mano
     la combinazione esatta).

### 4. `scripts/step5_demo.py` (nuovo)
4.1. Riusa lo stesso scenario di Step 4 (stella scelta s20, catalogo di
     riferimento). Esegue `simulate_collapse` due volte: una con
     `relativistic=False` (baseline, deve riprodurre i numeri già noti di
     Step 4: `t_collapse≈99.5 ms`), una con `relativistic=True`.
4.2. Stampa: tempo di collasso in entrambi i casi, compattezza massima
     raggiunta (shell più interna) nel caso relativistico, fattore di
     correzione massimo incontrato (`a_grav_GR/a_grav_Newtoniano` alla
     shell più interna, all'istante finale). Nessun plot Matplotlib
     (riservato a Step 7).
4.3. Disclaimer esplicito e obbligatorio nell'output: potenziale
     effettivo approssimato (Marek et al. 2006, "Case A"), non
     idrodinamica GR completa; il termine di pressione non è corretto;
     la classificazione finale del remnant tramite TOV arriva in Step 6.

### 5. `tests/test_relativistic.py` (nuovo)
5.1. Limite di campo debole: a compattezza e `P/(rho*c²)` piccoli (es.
     `r_cm` grande, `m_g`/`p_cgs` piccoli), verificare
     `abs(relativistic_grav_accel(...)/(-G_CGS*m_g/r_cm**2) - 1) <
     TOL_WEAK_FIELD_RTOL`.
5.2. Monotonia/segno: per una griglia di compattezze in (0, 0.8), il
     fattore di correzione (`|relativistic_grav_accel| /
     (G_CGS*m_g/r_cm**2)`) deve essere sempre >= 1 (con margine, es. >=
     1.0 - 1e-10 per tollerare rumore macchina al limite compattezza->0).
5.3. Guardia/evento: costruire uno stato con compattezza della shell più
     interna >= `COMPACTNESS_SAFETY_LIMIT`, verificare che
     `_schwarzschild_proximity_event` ritorni un valore <= 0 (evento
     scattato) e che `relativistic_grav_accel` sollevi `ValueError` se
     chiamata direttamente in quello stato.
5.4. **Regressione** (test più importante di questo step). **CORREZIONE
     (da review): `test_super_chandrasekhar_triggers_collapse` (Step 4)
     verifica solo proprietà qualitative — NON registra alcun valore
     numerico preciso di `t_collapse_s`/`rho_c_gcm3_t[-1]`. L'unico numero
     disponibile nel progetto è "≈99.5 ms" nel log di STATUS.md, con 3
     cifre significative, insufficiente per `rtol=1e-12`.** Procedura
     obbligatoria in DUE fasi, in quest'ordine:
     - **FASE A (PRIMA di modificare `collasso/dynamics.py`)**: con il
       codice di Step 4 ancora invariato, esegui `simulate_collapse` con
       gli stessi parametri di `test_super_chandrasekhar_triggers_collapse`
       (stella s20, `N_SHELLS_TEST_DEFAULT`, `t_max_s` calcolato come in
       quel test) e registra `t_collapse_s`, `collapse_reason`,
       `rho_c_gcm3_t[-1]` a PIENA precisione (tutte le cifre di un
       float). Questi numeri sono la baseline.
     - **FASE B (dopo aver implementato l'estensione)**: scrivi il test
       di regressione usando la baseline della FASE A come valore atteso
       (hardcoded nel test), confrontata con l'output di
       `simulate_collapse(..., relativistic=False)` sul codice ESTESO,
       tolleranza `rtol=1e-12`. Se la baseline fosse catturata DOPO aver
       già modificato `dynamics.py`, il test sarebbe tautologico (confronto
       contro se stesso) — esattamente ciò che questo test deve evitare.
5.5. Confronto fisico atteso: `simulate_collapse(..., relativistic=True)`
     per lo stesso caso s20 deve avere `t_collapse_s` MINORE (collasso più
     rapido) di `relativistic=False` — test qualitativo robusto (nessuna
     tolleranza quantitativa stretta, solo `<`), coerente con l'effetto
     fisico atteso (gravità effettiva più forte).

### 6. Chiusura del ciclo
6.1. `pytest tests/` (intera suite, Step 1-5) — nessuna regressione.
6.2. `scripts/step5_demo.py` eseguito, output verificato.
6.3. Passare la mano al `critic-fisico`: verificare la formula (limite di
     campo debole analitico, non solo test), il fatto che il termine di
     pressione non sia stato toccato (limite dichiarato), l'assenza di
     sconfinamento verso Step 6 (nessun TOV/classificazione remnant), e
     che il test di regressione (5.4) sia genuino (confronto contro numeri
     realmente prodotti da Step 4, non contro se stesso).
6.4. Orchestratore: eseguo io stesso `pytest`, scrivo il resoconto e
     aggiorno STATUS.md (salto optimizer/reporter come sub-agenti separati
     per restare lineare).
6.5. **STOP esplicito**, attesa conferma utente prima di Step 6.

## Piano ciclo corrente (Step 6)
Scritto direttamente dall'orchestratore (come Step 5, per uno svolgimento
lineare). A differenza di Step 5, questo step introduce un NUOVO schema
numerico (integrazione TOV) — vedi avviso di portata già condiviso con
l'utente in plan mode. Prima di fissare le costanti sotto, l'orchestratore
ha eseguito un calcolo di controllo indipendente (script Python ad-hoc,
non nel repository) risolvendo il sistema TOV per una griglia di densità
centrali: risultato `M_max≈0.82 Msun` (stesso ordine di grandezza del
valore storico di Oppenheimer & Volkoff 1939, ≈0.7 Msun — la fisica di
base è confermata prima di consegnare le costanti al coder). Il calcolo
di controllo ha anche rivelato un problema di overflow/instabilità
numerica vicino alla superficie (`dP/dx→0` per `x→0`, che rende
`dx/dr=dP/dr / dP/dx` stiff appena prima dell'evento terminale) — stessa
natura del clip già usato per `theta` in Lane-Emden (Step 2), da
applicare qui analogamente.

### Fisica (formule fissate, vedi anche il piano approvato in plan mode)
EOS di neutroni degeneri (stesso formalismo di Chandrasekhar/Step 3,
massa del neutrone al posto dell'elettrone, `n_n=rho/m_n` diretto, nessun
Ye — è l'EOS originale di Oppenheimer & Volkoff, Phys. Rev. 55, 374,
1939, gas di neutroni liberi non interagenti, limite dichiarato
esplicitamente: sottostima il vero limite osservativo ~2-2.2 Msun perché
trascura le interazioni nucleari):
- `x_n(rho) = (HBAR_CGS/(M_NEUTRON_G*C_LIGHT_CGS)) * (3*pi^2*rho/M_NEUTRON_G)^(1/3)`
- `P(x) = P0_n * chandrasekhar_f(x)` — **riuso diretto** di
  `collasso.eos.chandrasekhar_f` (f(x) è indipendente dalla massa del
  fermione, verificato dall'orchestratore nello script di controllo),
  `P0_n = M_NEUTRON_G^4*C_LIGHT_CGS^5/(24*pi^2*HBAR_CGS^3)`.
- `dP/dx = P0_n * 8*x^4/sqrt(1+x^2)` (stessa identità esatta di Step 3,
  riusata non ri-derivata).
- `rho(x) = M_NEUTRON_G^4*C_LIGHT_CGS^3*x^3/(3*pi^2*HBAR_CGS^3)`.

Equazioni TOV, parametrizzate in x (non rho/P, per riusare `dP/dx` chiuso
invece di invertire numericamente l'EOS ad ogni passo):
```
dP/dr = -G*(rho+P/c^2)*(m+4*pi*r^3*P/c^2) / (r^2*(1-2*G*m/(r*c^2)))
dm/dr = 4*pi*r^2*rho
dx/dr = (dP/dr) / (dP/dx)
```

### 1. Costanti (fissate qui, verificate dall'orchestratore col calcolo di
   controllo, non a discrezione del coder)
1.1. `M_NEUTRON_G = 1.67492749804e-24` (g, CODATA 2018) — nuova costante
     in `collasso/constants.py`.
1.2. `R0_CM_DEFAULT = 1.0` (cm — innesco vicino a r=0, verificato dare
     risultati stabili nel calcolo di controllo): `x(r0)≈x_c` (leading
     order, `dP/dr→0` per r→0 per simmetria — stesso principio di
     `theta'(0)=0` in Lane-Emden), `m(r0)≈(4/3)*pi*r0^3*rho_c`.
1.3. `X_FLOOR_RHS = 1e-6` — clamp INCONDIZIONATO applicato a `x` DENTRO il
     RHS (`x_clamped = max(x, X_FLOOR_RHS)`, valido anche se `x` è
     transitoriamente NEGATIVO durante una valutazione di prova del
     solver — non solo "vicino a zero da sopra"), prima di valutare
     `P(x)`/`dP/dx`. **Motivo (da review)**: `collasso.eos.chandrasekhar_f`
     (riusata direttamente) ha una guardia rigida che solleva `ValueError`
     per `x<0` — se il clamp non fosse un floor incondizionato, un
     piccolo overshoot negativo di `x` in un passo di prova di RK45
     farebbe esplodere lo stesso tipo di crash già risolto in Step 4/5
     (eccezione nel RHS non catturata da scipy). Stesso principio del
     clip `max(theta,0.0)` di Lane-Emden — mai un'eccezione nel RHS.
1.3bis. **(NUOVO, bloccante, da review)** `COMPACTNESS_TOV_CLAMP_MAX =
     0.99` — clamp INCONDIZIONATO sul termine di compattezza
     `2*G*m/(r*c^2)` dentro il RHS, PRIMA di valutare il denominatore
     `1-2*G*m/(r*c^2)` di `dP/dr` (`compactness_clamped =
     min(2*G*m/(r*c^2), COMPACTNESS_TOV_CLAMP_MAX)`), stesso pattern già
     collaudato in Step 5 (`COMPACTNESS_CLAMP_MAX`/`relativistic_grav_accel`).
     **Motivo**: il piano proteggeva solo `x` (superficie) ma non questo
     denominatore; la griglia esplora densità centrali (fino a
     `rho_c=1e17`) ben oltre il massimo di massa atteso (~3e15-1e16),
     cioè il ramo instabile/più compatto — se il denominatore si
     avvicinasse a zero o diventasse negativo durante una valutazione di
     prova del solver, `dP/dr` potrebbe cambiare segno silenziosamente
     (pressione che cresce verso l'esterno), rompendo la fisica senza che
     `sol.success` lo intercetti necessariamente. Il valore finale atteso
     della compattezza è basso (~0.16 per M_max≈0.82 Msun) — questa è una
     rete di sicurezza per le configurazioni intermedie esplorate dalla
     griglia, non per il risultato fisico atteso.
1.3ter. **(NUOVO, da review)** Nuovo test sintetico dedicato (stesso stile
     di `test_shell_crossing_event_and_v_floor`, Step 4): chiamare il RHS
     TOV direttamente con `x` negativo e con compattezza `>=
     COMPACTNESS_TOV_CLAMP_MAX`, verificare che non sollevi eccezioni e
     ritorni valori finiti.
1.4. `X_SURFACE_EVENT = 1e-4` (soglia dell'evento terminale di
     superficie, `direction=-1`, `terminal=True` — più permissiva di
     `X_FLOOR_RHS` per restare fuori dalla zona di stiffness prima che
     l'evento scatti, stesso principio del `xi0` di Lane-Emden).
1.5. `R_MAX_CM_DEFAULT = 5.0e7` (50000 km — ampio margine sopra qualunque
     raggio di stella di neutroni atteso, ~10-20 km, verificato nel
     calcolo di controllo).
1.6. `solve_ivp`: `method="RK45"`, `rtol=1e-8`, `atol=1e-6` (verificati
     dare risultati stabili nel calcolo di controllo), `max_step =
     R_MAX_CM_DEFAULT/2000` esplicito (stesso pattern di sicurezza già
     usato in `collasso.dynamics`, Step 4).
1.7. Check esplicito e incondizionato su `sol.success` (stesso pattern
     obbligatorio di Lane-Emden/Step 4), `RuntimeError` con `sol.message`
     se fallisce.
1.8. `RHO_C_GRID_GCM3 = np.logspace(14, 17, 30)` (g/cm³ — griglia di
     densità centrali per `find_ov_mass_limit`, verificata dal calcolo di
     controllo contenere il massimo, che cade fra 3e15 e 1e16 g/cm³).
1.9. **(NUOVO, consigliato da review)** Verifica di sensibilità a `r0`
     (non bloccante ma richiesta): ri-risolvere `rho_c=1e17` (l'estremo
     più severo della griglia) con `r0=0.1cm` e `r0=10cm` oltre al
     default `r0=1cm`, verificare che `M_msun` risultante cambi in modo
     trascurabile (es. <0.1% relativo) — sostituisce con un numero
     osservato l'attuale assunzione non verificata che l'innesco
     leading-order resti valido su tutta la griglia. Se il coder trova
     una sensibilità non trascurabile, deve fermarsi e riportarlo (stesso
     principio già applicato più volte in questo progetto), non ridurre
     `r0` unilateralmente senza motivarlo.
1.10. Tolleranza di validazione contro il valore storico (Oppenheimer &
     Volkoff 1939, ≈0.7 Msun): `M_TOV_SANITY_RANGE_MSUN = (0.4, 1.2)`.
     **Nota esplicita (a differenza della validazione stretta di Lane-Emden
     contro ξ1 di letteratura, Step 2)**: qui la tolleranza è
     deliberatamente larga, perché il progetto non riproduce la cifra
     esatta della letteratura, solo l'ordine di grandezza corretto.
     **AGGIORNAMENTO POST-VERIFICA (critic-fisico)**: l'ipotesi iniziale
     "lo scarto ~20% da 0.7 Msun è dovuto a dettagli di r0/soglia
     superficie/tolleranze del solver" è stata TESTATA e NON confermata:
     il critic-fisico ha scritto un secondo integratore TOV indipendente
     (metodo LSODA, r0 20x più piccolo, soglia di superficie 10x più
     permissiva) trovando `M_max=0.8428-0.8431 Msun`, entro ~0.03% dal
     valore del progetto (0.8429) — il risultato è quindi robusto ai
     dettagli numerici, non spiegabile con quelli. Lo scarto dal valore
     storico resta probabilmente dovuto a differenze fisiche più sottili
     (costanti CODATA aggiornate vs 1939, imprecisione del calcolo
     originale pre-computer) o a dettagli della griglia `rho_c`, non
     verificate ulteriormente qui — un confronto quantitativo più stretto
     è rimandato alla validazione contro GR1D (Step 8, già a piano). Il
     valore resta comunque entro il range di plausibilità dichiarato
     `M_TOV_SANITY_RANGE_MSUN=(0.4,1.2)` e l'ordine di grandezza è
     corretto — nessuna implicazione sulla correttezza del codice.

### 2. `collasso/eos_neutron.py` (nuovo, non modifica `eos.py`)
`fermi_x_neutron(rho_gcm3)`, `pressure_degenerate_neutron(rho_gcm3)`
(chiama `collasso.eos.chandrasekhar_f` — **verifica che il coder importi
davvero la funzione esistente, non la riscriva**), `dP_dx_neutron(x)`,
`rho_of_x_neutron(x)`. Stesso stile di validazione input di `collasso.eos`
(`rho_gcm3>0`, ecc.) per le funzioni pubbliche non sul path del RHS di
`solve_ivp`; le funzioni chiamate dentro `_tov_rhs` (vedi 3) non devono
mai sollevare eccezioni (stesso vincolo di `_shell_dvdt`/
`relativistic_grav_accel`).

### 3. `collasso/tov.py` (nuovo)
`TOVSolution` (dataclass: `rho_c_gcm3`, `r_cm`, `m_g`, `x`, `R_cm`,
`M_g`, `M_msun`). `solve_tov(rho_c_gcm3, r0_cm=R0_CM_DEFAULT,
r_max_cm=R_MAX_CM_DEFAULT, n_points=500) -> TOVSolution`: RHS con clamp
`X_FLOOR_RHS` (mai eccezioni), evento terminale a `X_SURFACE_EVENT`,
`max_step` esplicito, check `sol.success`. `find_ov_mass_limit(
rho_c_grid_gcm3=RHO_C_GRID_GCM3) -> (rho_c_max_gcm3, M_max_msun,
sequenza_completa)`: risolve `solve_tov` per ogni densità della griglia,
`np.argmax` sulla sequenza di masse (nessun raffinamento locale
richiesto, coerente con la tolleranza larga di 1.9).

### 4. `collasso/remnant.py` (nuovo, utility pura)
`classify_remnant(massa_msun, m_chandrasekhar_msun, m_tov_msun) -> str`:
`"white_dwarf"` se `massa_msun < m_chandrasekhar_msun`, `"neutron_star"`
se `m_chandrasekhar_msun <= massa_msun < m_tov_msun`, `"black_hole"` se
`massa_msun >= m_tov_msun`. Nessuna dipendenza da `solve_tov` (logica
pura, testabile in isolamento).

### 5. `scripts/step6_demo.py`
Calcola `find_ov_mass_limit()`, stampa `M_max_msun` e lo scarto
percentuale dal valore storico "~0.7 Msun" (Oppenheimer & Volkoff 1939).
Classifica il remnant per la stella scelta (s20): confronta
`massa_nucleo_msun` (1.50) con `chandrasekhar_mass_msun(0.46)` di Step 3
(1.2323 Msun) e col nuovo `M_max_msun` di questo step. Disclaimer
esplicito e obbligatorio: EOS di neutroni liberi non interagenti (nessuna
interazione nucleare forte), sottostima nota del limite osservativo reale
(~2-2.2 Msun, es. PSR J0740+6620). Nessun plot Matplotlib (Step 7).

### 6. `tests/test_tov.py` (nuovo)
- `M_max_msun` entro `M_TOV_SANITY_RANGE_MSUN`;
- `sol.success` (implicito, nessuna eccezione da `solve_tov`), `r_cm`
  crescente, `x` non-crescente (stesso stile di monotonia di Lane-Emden);
- `classify_remnant`: 3 casi (sotto Chandrasekhar, fra Chandrasekhar e
  TOV, sopra TOV), inclusi i valori di confine;
- test che `pressure_degenerate_neutron` richiami DAVVERO
  `collasso.eos.chandrasekhar_f` (import diretto, non una copia
  duplicata del codice — verificabile confrontando `chandrasekhar_f(x)`
  chiamato direttamente contro `pressure_degenerate_neutron(rho_of_x_neutron(x))
  / P0_n`, che devono coincidere esattamente).

### 7. Chiusura del ciclo
7.1. `pytest tests/` (intera suite) — nessuna regressione.
7.2. `scripts/step6_demo.py` eseguito, output verificato.
7.3. Passare la mano al `critic-fisico`: EOS neutronica dichiarata
     onestamente come semplificata, riuso genuino di `chandrasekhar_f`,
     valore plausibile rispetto al riferimento storico, nessuno
     sconfinamento (niente rotazione/campo magnetico/interazioni
     nucleari introdotte "di nascosto").
7.4. Orchestratore: `pytest`, resoconto, aggiornamento STATUS.md (salto
     optimizer/reporter come in Step 5).
7.5. **STOP esplicito**, attesa conferma utente prima di Step 7.

### GIORNO 6 — 2026-08-11 — Step 6 completato
**Implementato**: `collasso/eos_neutron.py` (nuovo) — EOS di neutroni
degeneri liberi non interagenti, formalismo originale di Oppenheimer &
Volkoff (1939), riuso genuino di `collasso.eos.chandrasekhar_f` (import
diretto, verificato dal critic-fisico leggendo il codice — nessuna
duplicazione). `collasso/tov.py` (nuovo) — integrazione delle equazioni
TOV parametrizzate nel parametro di Fermi x (per riusare `dP/dx` in forma
chiusa da Step 3), innesco vicino a r=0 (stesso principio del `xi0` di
Lane-Emden, verificato con sensibilità a r0 trascurabile: scarto
~3e-7% fra r0=0.1/1/10 cm), due clamp incondizionati nel RHS (su x e
sulla compattezza, mai un'eccezione — stesso pattern già stabilito in
Step 4/5), evento terminale di superficie. `collasso/remnant.py` (nuovo)
— classificazione pura (nana bianca / stella di neutroni / buco nero).

**Risultato fisico**: limite di massa OV calcolato = **0.8429 Msun**
(stesso ordine di grandezza del valore storico di Oppenheimer & Volkoff
1939, ~0.7 Msun — scarto 20.4%, confermato NON dovuto a dettagli di
implementazione dal secondo integratore indipendente del critic-fisico,
vedi sotto). Per la stella scelta (s20): `massa_nucleo_msun=1.50 Msun` >
`chandrasekhar_mass_msun(0.46)=1.2323 Msun` > `M_TOV=0.8429 Msun` →
classificazione **"black_hole"**, con disclaimer esplicito che con
un'EOS nucleare realistica (non implementata, limite dichiarato) questo
stesso nucleo diventerebbe verosimilmente una stella di neutroni, non un
buco nero — la classificazione riflette la sottostima nota dell'EOS
semplificata, non è una previsione fisica affidabile per questo
progenitore specifico.

**Test**: `pytest tests/` (intera suite) → **61/61 passati**, 0 falliti
(55 da Step 1-5 + 6 nuovi di Step 6). Nessuna regressione.

**Verifica critic-fisico**: verdetto **"conforme con riserve"** (riserve
minori, corrette dall'orchestratore dopo la verifica). Il critic-fisico
ha scritto un SECONDO integratore TOV indipendente da zero (metodo
LSODA, r0 20 volte più piccolo, soglia di superficie 10 volte più
permissiva) trovando `M_max=0.8428-0.8431 Msun`, entro ~0.03% dal
valore del progetto — forte corroborazione indipendente della
correttezza implementativa. Ha però evidenziato che la spiegazione
iniziale dello scarto dal valore storico ("dettagli di r0/soglia/
solver") non regge a questo cross-check (il risultato è robusto a
quei dettagli) — corretto in STATUS.md, verifica quantitativa più
stretta rimandata a Step 8 (GR1D). Confermati: entrambi i clamp
incondizionati nel RHS, riuso genuino di `chandrasekhar_f`, coerenza
interna della classificazione, perimetro fisico rispettato (EOS
dichiarata onestamente come semplificata, nessuna interazione nucleare
"nascosta", nessuna rotazione/campo magnetico).

**Stato progetto**: 6/9 step completati.
**Prossimo step proposto**: Step 7 — Pipeline di visualizzazione
(Matplotlib: animazione + grafici numerici). In attesa di conferma
esplicita dell'utente prima di procedere.

### GIORNO 5 — 2026-08-11 — Step 5 completato
**Implementato**: `collasso/relativistic.py` (nuovo) — correzione
relativistica approssimata al termine di gravità, potenziale effettivo
"Case A" (Marek, Dimmelmeier, Janka, Müller & Buras 2006, A&A 445, 273):
`a_grav_GR = -G*[m+4πr³P/c²]*[1+P/(ρc²)] / (r²*[1-2Gm/(rc²)])`, riduce
esattamente a `-G*m/r²` in campo debole. `collasso/dynamics.py` esteso
(non riscritto) con parametro `relativistic: bool = False`,
retrocompatibile per costruzione (con `relativistic=False` il
comportamento è bit-identico a Step 4, verificato da un test di
regressione con baseline catturata PRIMA della modifica); nuovo evento
terminale di prossimità al raggio di Schwarzschild (soglia di
compattezza 0.9, monitorata su TUTTE le shell, non solo la più interna),
clamp numerico di sicurezza (`COMPACTNESS_CLAMP_MAX=0.99`, mai
un'eccezione nel RHS di `solve_ivp` — stesso principio di `V_FLOOR_CM3`
di Step 4). `scripts/step5_demo.py`: per la stella scelta (s20), il
caso relativistico collassa il **4.49% più rapidamente** del caso
Newtoniano (94.99 ms vs 99.45 ms), compattezza massima raggiunta
4.12e-2, coerente con l'effetto fisico atteso (gravità effettiva più
forte).

**Percorso del ciclo (deliberatamente più corto di Step 4)**: per
richiesta esplicita dell'utente di uno svolgimento lineare, la
formalizzazione del piano e la chiusura del ciclo sono state scritte
direttamente dall'orchestratore (non dai sub-agenti `planner`/
`optimizer`/`reporter`), riducendo la pipeline a due sole chiamate:
`reviewer` (un giro, ha trovato 3 problemi concreti e corretti — vedi
sotto) e `coder`+`critic-fisico`. Nessun nuovo studio di convergenza
numerica è stato necessario (a differenza di Step 4): la correzione è
una formula chiusa locale per shell, non un nuovo schema di
discretizzazione.

**Correzioni dal reviewer** (applicate prima dell'implementazione): (a)
tolleranza del test di campo debole incoerente con l'esempio dato
(corretta: compattezza/P/(ρc²) ~1e-8, tolleranza 1e-5); (b) guardia di
compattezza che avrebbe sollevato un'eccezione nel RHS di `solve_ivp`
(stesso errore già corretto in Step 4 — sostituita con clamp numerico);
(c) evento di prossimità al raggio di Schwarzschild esteso a tutte le
shell, non solo la più interna; (d) procedura in due fasi per il test di
regressione, per evitare un confronto tautologico.

**Scoperte del coder durante l'implementazione** (documentate, non
decise in silenzio): (a) correzione di un errore di segno nel testo
letterale del piano (`grav = relativistic_grav_accel(...)`, senza
negazione aggiuntiva — la formula include già il segno); (b)
identificata una contraddizione fra due punti del piano (mai eccezioni
vs. test che si aspettava un'eccezione), risolta dando priorità al
vincolo di sicurezza numerica (mai eccezioni nel RHS), con la
contraddizione segnalata esplicitamente nel codice.

**Test**: `pytest tests/` (intera suite) → **55/55 passati**, 0 falliti
(50 da Step 1-4 + 5 nuovi di Step 5). Nessuna regressione.

**Verifica critic-fisico**: verdetto **"conforme con riserve"** (riserve
minori, non bloccanti): un test non esercitava realmente il clamp
numerico (testava una compattezza ancora fisicamente valida) — corretto
dall'orchestratore dopo la verifica (ora testa compattezza=1.5); naming
ambiguo di una variabile nel demo (`fattore_correzione_max` misurava in
realtà solo la shell più interna, non il massimo su tutte le shell) —
rinominata. Confermate formule, segno, limite di campo debole (scarto
osservato 3.0e-8, ben entro tolleranza), monotonia del fattore di
correzione (sempre ≥1), perimetro fisico rispettato (termine di
pressione non toccato, nessun TOV/neutrini/rotazione introdotto),
plausibilità quantitativa del risultato (-4.49% scomposto e verificato
termine per termine).

**Stato progetto**: 5/9 step completati.
**Prossimo step proposto**: Step 6 — Check limite TOV, classificazione
remnant. In attesa di conferma esplicita dell'utente prima di procedere.

## Retrofit Step 6 — EOS realistica (piecewise polytrope, Read et al. 2009)
Richiesto esplicitamente dall'utente dopo la chiusura di Step 6: sostituire
il gas di neutroni liberi (M_max=0.8429 Msun, scarto 20% dal valore
storico) con la parametrizzazione a politropi a tratti di Read, Lackey,
Owen & Friedman (2009, PRD 79, 124032, arXiv:0812.2163), fit a SLy/APR4,
accurata entro ~1% sulle relazioni massa-raggio.

**Verifica preliminare dell'orchestratore** (prima di scrivere questo
piano): algoritmo recuperato dal codice sorgente di produzione di
LALSuite (LIGO/Virgo,
`lalsimulation/lib/LALSimNeutronStarEOSPiecewisePolytrope.c`, che
implementa letteralmente Read et al. 2009), più uno script di controllo
indipendente in Python. Risultato: **SLy → M_max=2.000 Msun, APR4 →
M_max=2.187 Msun**, entro il 2-3% dei valori di letteratura (SLy≈2.05,
APR4≈2.20 Msun) — verificato PRIMA di consegnare le costanti al coder.

### Algoritmo (fissato, non a discrezione del coder)
**Crosta** (fit fisso a SLy4, Read et al. Tabella II, uguale per
qualunque EOS ad alta densità, trascritta da LALSuite):
```
RHO_LOW_CGS = [0, 2.44033979e7, 3.78358138e11, 2.62780487e12]   # g/cm^3
K_LOW_CGS   = [6.11252036792443e12, 9.54352947022931e14,
               4.787640050002652e22, 3.593885515256112e13]
GAMMA_LOW   = [1.58424999, 1.28732904, 0.62223344, 1.35692395]
```
**Nucleo** (3 politropi, densità di separazione fisse
`RHO_1_CGS = 10**14.7`, `RHO_2_CGS = 10**15.0`):
```
SLY  = {log10_p1: 34.348, gamma1: 3.005, gamma2: 2.988, gamma3: 2.851}
APR4 = {log10_p1: 34.269, gamma1: 2.830, gamma2: 3.445, gamma3: 3.348}
```
**Costruzione** (7 pezzi per SLy/APR4 — il caso "ponte" a 8 pezzi di
LALSuite, per parametri fuori range, NON serve qui, verificato dallo
script di controllo; se `rho0` non cade fra `RHO_LOW_CGS[3]` e
`RHO_1_CGS`, il coder deve sollevare un errore esplicito, non gestire
silenziosamente il caso non implementato):
```
p1 = 10**log10_p1
k1 = p1 / RHO_1_CGS**gamma1
k2 = p1 / RHO_1_CGS**gamma2
k3 = k2 * RHO_2_CGS**(gamma2-gamma3)
rho0 = (K_LOW_CGS[3]/k1) ** (1/(gamma1-GAMMA_LOW[3]))
```
`rho_boundaries = [0, RHO_LOW[1], RHO_LOW[2], RHO_LOW[3], rho0,
RHO_1_CGS, RHO_2_CGS]`, `K = [K_LOW..., k1, k2, k3]`,
`Gamma = [GAMMA_LOW..., gamma1, gamma2, gamma3]`.

Per ciascun pezzo i: `n_i = 1/(Gamma_i-1)`, `P_i = K_i *
rho_boundary_i^Gamma_i` (P al bordo 0 = 0). Costanti di integrazione per
l'energia (continuità di ε, derivazione dalla prima legge della
termodinamica, riderivata e verificata indipendentemente
dall'orchestratore contro la versione geometrizzata di LALSuite):
```
a_0 = 0
a_i = a_(i-1) + (n_(i-1)-n_i) * P_i / (rho_boundary_i * c^2)   per i>=1
```
**Valutazione EOS** (pezzo attivo i = indice del `rho_boundary` più
grande <= rho):
```
P(rho) = K_i * rho^Gamma_i
epsilon(rho) = (1+a_i)*rho*c^2 + n_i*P(rho)     # erg/cm^3
dP_drho(rho) = K_i * Gamma_i * rho^(Gamma_i-1)  # forma chiusa
```
**Nota fisica — CORREZIONE POST-REVIEW, scoperta importante**: il
reviewer ha segnalato che l'affermazione "Step 6 usava rho, approssimazione
tollerabile" non era quantificata e rischiava di essere sbagliata in
ordine di grandezza vicino al picco OV, dove il gas è relativistico.
L'orchestratore ha VERIFICATO con un calcolo di controllo indipendente,
risolvendo la STESSA EOS a neutroni liberi di Step 6 sia con `rho` (come
implementato) sia con `epsilon/c^2` (formula esatta derivata sotto):
```
rho (Step 6, come implementato):  M_max = 0.8427 Msun
epsilon/c^2 (fisicamente corretto): M_max = 0.7100 Msun
```
**Lo scarto "inspiegato" del 20% dal valore storico di Oppenheimer &
Volkoff (~0.7 Msun), documentato in STATUS.md come "probabilmente dovuto
a differenze fisiche sottili non verificate", era in realtà interamente
questo bug** — con la sorgente corretta, il risultato è entro l'1.4% dal
valore storico. Questo NON è un problema del politropo a tratti: è un
bug preesistente in `collasso/tov.py` di Step 6 (`dm/dr = 4*pi*r^2*rho`
usa la densità di massa a riposo invece della densità di energia totale,
in entrambe le equazioni TOV) che va corretto ANCHE nel path
`eos=None` (gas di neutroni liberi), non solo introdotto per il nuovo
politropo a tratti.

**Formula per l'energia del gas di Fermi libero** (derivata
dall'orchestratore dalla relazione termodinamica esatta
`epsilon = n*mu - P` con `mu = m*c^2*sqrt(1+x^2)` — energia di Fermi
relativistica incluso il riposo — e verificata nel limite non-relativistico
x->0, dove si riduce correttamente a `epsilon -> rho*c^2`):
```
epsilon(x) = P0_n * 3*[x*(2*x^2+1)*sqrt(1+x^2) - asinh(x)]
```
(stesso prefattore `P0_n` già usato per la pressione).

**Conseguenza per il piano**: `collasso/eos_neutron.py` guadagna
`energy_density_of_x_neutron(x)`/`energy_density_of_rho_neutron(rho)`
(formula sopra). `collasso/tov.py`, il path ESISTENTE `eos=None` (Step 6)
va CORRETTO per usare questa densità di energia come sorgente in
`dP/dr` e `dm/dr`, non solo il nuovo path piecewise-polytrope — è la
correzione di un bug fisico reale, non un cambiamento di comportamento
arbitrario. Questo rompe necessariamente la nozione originale di "test di
regressione bit-identico a Step 6" (sezione File coinvolti, `tests/test_tov.py`,
sotto corretta di conseguenza): il path `eos=None` ora dà `M_max≈0.71
Msun` invece di `0.8429` — un CAMBIAMENTO INTENZIONALE, documentato,
motivato, che va segnalato chiaramente nel log come "bug fisico corretto
retroattivamente", non nascosto dietro una preservazione forzata del
vecchio numero.

**Tre confronti da mostrare nel demo (invece di due)**, per disaggregare
onestamente i due effetti distinti (fix del bug vs. fisica nucleare
reale), esattamente come richiesto dal reviewer:
1. Step 6 originale (bug, per riferimento storico): 0.8429 Msun
2. Gas di neutroni liberi CORRETTO (epsilon/c^2, valida l'integratore TOV
   contro il valore storico O&V 1939 ≈0.7 Msun): 0.7100 Msun
3. SLy/APR4 (fisica nucleare realistica): ~2.0-2.2 Msun

**TOV riparametrizzato in rho** (generalizzazione diretta del `dx/dr` di
Step 6):
```
dP/dr = -G*(epsilon/c^2+P/c^2)*(m+4*pi*r^3*P/c^2) / (r^2*(1-2*G*m/(r*c^2)))
drho/dr = (dP/dr) / dP_drho(rho)
dm/dr = 4*pi*r^2*epsilon/c^2
```
Stessi due clamp incondizionati già stabiliti in Step 6 (mai eccezioni
nel RHS): floor su rho (`RHO_FLOOR_RHS_CGS=1e-3`), clamp compattezza
(`COMPACTNESS_TOV_CLAMP_MAX=0.99`, riusato identico). Evento terminale
quando rho scende sotto `RHO_SURFACE_EVENT_CGS=1.0` g/cm³.

### File coinvolti
- `collasso/eos_neutron.py` (Step 6, ESTESO — non solo il nuovo modulo):
  nuova `energy_density_of_x_neutron(x)`/`energy_density_of_rho_neutron(rho)`
  con la formula `epsilon(x)` sopra, verificata dall'orchestratore.
- `collasso/eos_piecewise_polytrope.py` (nuovo): costanti sopra (citazioni
  esplicite Read et al. 2009 + LALSuite come fonte di verifica incrociata),
  `PiecewisePolytropeEOS` (dataclass), `build_sly()`, `build_apr4()`,
  `pressure_of_rho`, `energy_density_of_rho`, `dP_drho`.
- `collasso/tov.py` (esteso, **CORREZIONE DI BUG sul path esistente
  `eos=None`, non solo aggiunta**): sia il path storico gas-libero
  (`eos=None`) sia il nuovo path piecewise-polytrope devono usare la
  densità di ENERGIA come sorgente in `dP/dr` e `dm/dr` — il vecchio
  comportamento (rho) resta SOLO come riferimento storico/didattico
  esplicitamente etichettato "non fisicamente corretto, preservato per
  confronto", non più il calcolo effettivo di `eos=None`. Nuova griglia
  `RHO_C_GRID_REALISTIC_CGS = np.logspace(14.5, 16, 30)` per le EOS
  realistiche (picco SLy verificato ~2e15 g/cm³, picco APR4 verificato
  dall'orchestratore ~1.87e15 g/cm³ — entrambi comodamente dentro la
  griglia).
- `scripts/step6_demo.py` (aggiornato): **TRE confronti** (non due), per
  disaggregare onestamente il fix del bug dalla fisica nucleare: (1)
  Step 6 originale/bug, gas libero con `rho`, per riferimento storico:
  0.8429 Msun; (2) gas libero CORRETTO con `epsilon/c^2`: ~0.71 Msun,
  valida l'integratore TOV stesso contro Oppenheimer & Volkoff 1939
  (~0.7 Msun, ora entro ~1-2%, non più 20%); (3) SLy/APR4 (fisica
  nucleare realistica): ~2.0-2.2 Msun, confrontati con letteratura
  (2.05/2.20 Msun) e con PSR J0740+6620=2.08±0.07 Msun (Fonseca et al.
  2021). Riclassifica s20 con SLy e APR4. Disclaimer aggiornato: SLy/APR4
  sono fit fenomenologiche (non EOS nucleari complete) accurate entro
  ~1% sulle relazioni massa-raggio.
- `tests/test_eos_piecewise_polytrope.py` (nuovo): continuità di P e
  epsilon a ogni confine (7), per SLy e APR4; riproduzione esatta dei
  parametri di tabella; `dP_drho` vs differenza finita (cross-check).
- `tests/test_tov.py` (esteso): `M_max_msun` per SLy entro ±10% di 2.05
  Msun, APR4 entro ±10% di 2.20 Msun; **nuovo test per il gas libero
  CORRETTO** (`eos=None`, ora con epsilon/c^2) entro ±10% del valore
  storico 0.7 Msun — sostituisce il vecchio concetto di "regressione
  bit-identica a Step 6" (non più applicabile: il bug fix cambia
  intenzionalmente il numero, da 0.8429 a ~0.71, un miglioramento
  documentato, non una regressione da evitare).
- **(NUOVO, da review)** Controllo di sensibilità alle tolleranze del
  solver: il coder deve verificare che dimezzare `rtol`/`atol` non cambi
  `M_max` in modo apprezzabile per SLy (le 6 discontinuità di `dP_drho`
  ai confini fra pezzi polytropici sono un fatto noto e atteso — C0 ma
  non C1 — non un errore, ma vanno verificate empiricamente, stesso
  principio già applicato alla sensibilità di `r0` in Step 6).

### Pipeline
1. **coder**: implementa (incluso il fix del bug su `eos=None`),
   verifica di persona i numeri di controllo (incluso il check di
   sensibilità alle tolleranze) prima di consegnare. Il reviewer ha già
   verificato le formule algebriche (continuità a_i, dP_drho, rho0) e
   confermato tutte corrette con calcolo indipendente — non serve un
   secondo giro di reviewer prima del coder, dato che il gap principale
   trovato (rho vs epsilon/c^2) è già stato investigato e quantificato
   dall'orchestratore sopra.
2. **critic-fisico**: verifica formule/continuità leggendo il codice,
   plausibilità di M_max contro letteratura E contro PSR J0740+6620,
   correttezza del fix rho->epsilon/c^2 anche sul path storico,
   nessuno sconfinamento (niente rotazione/campo magnetico/temperatura
   finita).
3. Orchestratore: `pytest`, resoconto, aggiornamento STATUS.md (CLAUDE.md
   già aggiornato dall'orchestratore prima di questa sezione).
4. **STOP esplicito**, attesa conferma utente prima di Step 7.

### GIORNO 6bis — 2026-08-11 — Retrofit Step 6 completato (EOS realistica)
**Implementato**: `collasso/eos_piecewise_polytrope.py` (nuovo) — EOS a
politropi a tratti di Read, Lackey, Owen & Friedman (2009, PRD 79,
124032), crosta fissa (fit SLy4, 4 pezzi, trascritta da LALSuite e
verificata cifra per cifra dal critic-fisico contro il piano) + 3
politropi ad alta densità (parametri SLy e APR4 da Tabella III), 7 pezzi
totali con continuità di pressione e densità di energia (costanti `a_i`)
verificate algebricamente e numericamente. `collasso/eos_neutron.py`
esteso con `energy_density_of_x_neutron`. `collasso/tov.py`: **corretto
un bug fisico preesistente in Step 6** (il TOV usava densità di massa a
riposo invece di densità di energia totale come sorgente, sia in dP/dr
sia in dm/dr) — il vecchio comportamento è preservato come funzioni
`*_legacy_bug_rho_source` esplicitamente etichettate, mai usate per la
classificazione di produzione.

**Risultati fisici** (tre confronti, per disaggregare onestamente il fix
del bug dalla fisica nucleare, come richiesto dal reviewer):
1. Step 6 originale/bug (riferimento storico): 0.8429 Msun
2. Gas di neutroni liberi CORRETTO: **0.7100 Msun** — scarto **1.4%**
   dal valore storico di Oppenheimer & Volkoff 1939 (~0.7 Msun), contro
   il 20.4% di Step 6 — lo scarto "inspiegato" documentato nel log di
   Step 6 era interamente questo bug, non differenze fisiche sottili
   come ipotizzato allora.
3. **SLy: 2.0020 Msun** (scarto 2.3% da letteratura ≈2.05 Msun), **APR4:
   2.1872 Msun** (scarto 0.6% da letteratura ≈2.20 Msun). Confronto
   osservativo: PSR J0740+6620 = 2.08±0.07 Msun (Fonseca et al. 2021) —
   SLy a 1.1σ, APR4 a 1.5σ.

Per la stella scelta (s20, massa_nucleo_msun=1.50 Msun): la
classificazione remnant cambia da **"black_hole"** (gas libero, M_TOV
0.71-0.84 Msun) a **"neutron_star"** (SLy e APR4, M_TOV 2.0-2.2 Msun) —
il miglioramento di realismo richiesto dall'utente si riflette
direttamente nella classificazione finale.

**Test**: `pytest tests/` (intera suite) → **81/81 passati**, 0 falliti
(80 dal coder + 1 test aggiuntivo dell'orchestratore, vedi sotto).
Controllo di sensibilità alle tolleranze del solver (dimezzando
rtol/atol per SLy): scarto relativo su M_max = 7.5e-8, trascurabile —
le discontinuità C0 di dP/drho ai confini fra pezzi (attese, non un
errore) non causano instabilità.

**Verifica critic-fisico**: verdetto **"conforme con riserve"** (riserve
minori). Ha rifatto un secondo calcolo TOV indipendente da zero
(integratore RK4 a passo fisso, inversione P→ρ via `brentq`, nessun
riuso del codice del progetto) per il gas libero corretto: **0.7093
Msun** contro 0.7100 dichiarato, scarto ~0.1% — forte corroborazione
indipendente. Confermati: crosta trascritta cifra per cifra corretta,
fix rho→epsilon/c² applicato correttamente in entrambe le equazioni per
entrambi i path, path legacy isolato e mai usato in produzione,
coerenza interna della riclassificazione di s20, perimetro fisico
rispettato (nessuna rotazione/campo magnetico/temperatura finita).
Riserva segnalata (non verificabile dal critic-fisico, ambiente senza
accesso a Internet): confronto diretto delle costanti di crosta/nucleo
contro la fonte bibliografica primaria — già coperto dall'orchestratore
PRIMA di scrivere il piano (fetch diretto del codice sorgente di
produzione LALSuite/LIGO, più script di controllo indipendente che ha
riprodotto i valori di letteratura entro 2-3%). Seconda riserva (mancava
un test che esercitasse entrambi i clamp insieme nel path
piecewise-polytrope) risolta dall'orchestratore aggiungendo
`test_tov_rhs_piecewise_never_raises_for_negative_rho_and_high_compactness`
in `tests/test_tov.py` (suite ri-verificata: 81/81).

**CLAUDE.md**: tabella vincoli aggiornata (riga "Classificazione
remnant") per riflettere la scelta EOS piecewise polytrope Read et al.
2009, con nota che il gas di neutroni liberi resta disponibile come
caso limite storico/didattico.

**Stato progetto**: 6/9 step completati (invariato — questo è un
retrofit di Step 6, non un nuovo step).
**Prossimo step proposto**: Step 7 — Pipeline di visualizzazione
(Matplotlib: animazione + grafici numerici). In attesa di conferma
esplicita dell'utente prima di procedere.

## Piano ciclo corrente (Step 7)
Richiesto dall'utente: un ENTRY POINT UNICO (non demo separati) che
chieda la stella in input e produca insieme output numerico finale
(velocità, energie, tempi scala, classificazione remnant) e animazione
grafica del collasso, con TUTTI i disclaimer fisici rilevanti mostrati
esplicitamente nell'output finale (non solo in log/STATUS.md). Rischio
più basso di Step 4/6 (nessuna nuova discretizzazione: compone funzioni
già validate + Matplotlib, mai usato finora nel progetto).

### 1. `collasso/pipeline.py` (nuovo) — orchestrazione pura, senza I/O
`run_full_simulation(star_id: str) -> SimulationResult`:
1. Carica il catalogo (`collasso.catalog.load_reference_catalog`,
   `get_progenitor_by_id` — errore chiaro se `star_id` non esiste).
2. Profilo di equilibrio iniziale: `collasso.lane_emden.solve_lane_emden`
   + `physical_profile` (Step 2), `n=stella.n_politropico`.
3. Dinamica: `collasso.dynamics.build_initial_shells` +
   `simulate_collapse` (Step 4/5), `N_SHELLS=200` (stesso default degli
   script demo precedenti), `t_max_s = T_MAX_FREE_FALL_MULTIPLIER *
   free_fall_time_s(stella.densita_centrale_gcm3)`, **`relativistic=True`
   di default** (scelta esplicita, dichiarata nei disclaimer, non
   nascosta).
4. Energie (formule chiuse, nessuna nuova fisica):
   - `KE_finale_erg = sum(0.5*delta_m_g_i*v_cms_i(t_finale)**2)`.
   - Autoenergia gravitazionale discretizzata delle shell — **CORREZIONE
     da review**: usare la massa RACCHIUSA A META' SHELL (non il bordo
     esterno incluso), per ridurre il bias sistematico da somma di
     Riemann a estremo destro (O(1/N), sovrastima |PE|) a un errore
     O(1/N²) simile a una regola del punto medio:
     ```
     m_mid_i = m_enclosed_g_i - delta_m_g_i/2
     PE_erg(t) = -G_CGS * sum(m_mid_i * delta_m_g_i / r_cm_i(t))
     ```
     calcolata a `t=0` e `t=t_finale` (o all'ultimo istante disponibile
     se il collasso non converge entro `t_max_s` — vedi punto 7 sotto);
     `Delta_PE_erg = PE_finale - PE_iniziale` (atteso negativo: energia
     liberata). Documentare nel docstring che resta un'approssimazione
     discreta (non esatta), non validata contro un caso analitico
     (a differenza di Lane-Emden/TOV altrove nel progetto) — dichiararlo
     onestamente come tale, non presentarla come precisa.
   - **Test minimo di sanità richiesto (da review)**: `tests/test_pipeline.py`
     deve verificare che `PE_erg < 0` sempre, e che l'ordine di grandezza
     sia plausibile (`abs(PE_erg)` confrontabile con `G*M_tot^2/R_tipico`
     entro un fattore ~2-5, non un numero a caso).
   - **Limite dichiarato**: bilancio energetico include SOLO cinetica +
     autoenergia gravitazionale, NON energia interna/termica né perdite
     per neutrini — `Delta_PE_erg + KE_finale_erg` non deve essere letto
     come verifica di conservazione dell'energia (il bilancio non è
     chiuso per costruzione, coerente col limite "nessun raffreddamento
     neutrinico" già dichiarato dal progetto). Va scritto esplicitamente
     nel docstring E nei disclaimer.
5. Classificazione remnant: `collasso.eos.chandrasekhar_mass_msun(ye)` +
   `collasso.tov.find_ov_mass_limit(RHO_C_GRID_REALISTIC_CGS, eos=...)`
   per **SLy e APR4** (Retrofit Step 6 — mai il path storico/bacato
   `*_legacy_bug_rho_source`, che resta riservato ai soli script/test
   storici) + `collasso.remnant.classify_remnant` per entrambe le EOS.
6. `SimulationResult` (dataclass): tutti i campi sopra (stella, profilo
   iniziale, `CollapseSolution` completa, energie, masse di soglia,
   classificazioni) + `disclaimers: list[str]` costruita QUI (non nello
   script), per essere testabile indipendentemente da come viene poi
   stampata.

**Contenuto esatto di `disclaimers`** (**8 voci**, non 7 — aggiunta la
simmetria sferica su segnalazione del reviewer: è un vincolo di modello
esplicito in CLAUDE.md, "Dinamica del collasso | Shell Lagrangiane,
simmetria sferica", e probabilmente il limite concettualmente più
importante per un utente finale — un collasso reale è 3D, con
convessione/instabilità SASI che nei modelli 1D sono spesso proprio
ciò che permette/impedisce l'esplosione. Testo libero ma deve contenere
sottostringhe verificabili nei test — il coder scrive il testo, i test
verificano parole chiave):
1. Catalogo placeholder (Step 1) — parola chiave: "placeholder".
2. Mismatch noto catalogo/Lane-Emden (Step 1/2) — parola chiave:
   "mismatch" o "incoeren".
3. Nessun raffreddamento neutrinico (Step 4+) — parola chiave:
   "neutrini".
4. Nessuna viscosità artificiale (Step 4) — parola chiave: "viscosità"
   o "viscosita".
5. Correzione relativistica approssimata "Case A" (Step 5) — parola
   chiave: "Case A" o "approssimat" + "relativistic" o "GR".
6. EOS SLy/APR4 fenomenologiche, non EOS a molti corpi complete
   (Retrofit Step 6) — parola chiave: "fenomenolog".
7. Nessuna rotazione/campo magnetico — parola chiave: "rotazione".
8. **(NUOVO, da review)** Simmetria sferica (1D): il collasso reale è
   intrinsecamente tridimensionale (convezione, instabilità SASI); il
   modello 1D non può catturare questi effetti, potenzialmente
   determinanti per l'esito reale (esplosione vs formazione diretta del
   remnant) — parola chiave: "simmetria sferica" o "1D".

### 2. `collasso/visualization.py` (nuovo)
`import matplotlib; matplotlib.use("Agg")` **PRIMA** di qualunque altro
import di matplotlib (obbligatorio per evitare errori headless su
sistemi senza display — va fatto in testa al modulo, non dentro le
funzioni).
- `animate_collapse(result, output_path)`: `FuncAnimation`, un frame =
  raggio (km) di ogni shell vs frazione di massa racchiusa
  (`m_enclosed_g/M_tot`), a un dato istante; titolo con tempo corrente
  (ms) e densità centrale corrente. Sottocampiona a
  `N_FRAMES_MAX=100` frame se `CollapseSolution.t_s` ne contiene di più
  (`np.linspace` di indici, non slicing arbitrario che salti l'istante
  finale). Salvata con `PillowWriter` (GIF, nessuna dipendenza ffmpeg).
- `plot_summary(result, output_path)`: figura statica 3 pannelli
  (`plt.subplots(1,3)` o `2x2` con un pannello vuoto): (a) densità
  centrale vs tempo (scala log su y), (b) velocità della shell più
  interna vs tempo, (c) curva massa-raggio TOV per SLy e APR4 (usando
  `find_ov_mass_limit` con la griglia esistente, riusando i punti già
  calcolati in `SimulationResult` per non ricalcolare) con un marker
  sulla massa della stella scelta.
- Entrambe le funzioni creano le directory intermedie di `output_path`
  se non esistono (`Path(output_path).parent.mkdir(parents=True,
  exist_ok=True)`) e ritornano il path assoluto scritto.

### 3. `scripts/run_simulation.py` (nuovo — L'ENTRY POINT UNICO)
- `sys.argv[1]` se presente = `star_id` (uso non interattivo/test);
  altrimenti stampa l'elenco stelle disponibili dal catalogo e chiede
  `input("Seleziona una stella: ")`, con validazione e nuovo prompt (max
  3 tentativi, poi errore chiaro) se l'id non esiste.
- Chiama `run_full_simulation`, stampa l'output numerico strutturato
  (sezioni: Stella, Profilo iniziale, Dinamica, Energie con nota
  esplicita sul bilancio non chiuso, Classificazione remnant per
  entrambe le EOS).
- **(NUOVO, da review) Sezione "Dinamica" — gestione esplicita di
  `collapsed=False`**: stampa SEMPRE `collapsed`, `collapse_reason`,
  `t_collapse_s` (anche se `None`). Se `collapsed=True`: framing "collasso
  avvenuto a t=... (motivo: ...)". Se `collapsed=False`: framing diverso
  e esplicito, es. "collasso NON completato entro t_max_s — l'integrazione
  si è fermata al tempo massimo senza che nessun evento fisico scattasse;
  le energie/densità riportate sotto sono all'ULTIMO istante disponibile,
  non a un vero momento di collasso" — per non far credere a un utente
  che il collasso sia avvenuto quando l'integrazione si è solo fermata.
- **Sezione dedicata "LIMITI DEL MODELLO" con tutti gli 8 disclaimer**,
  ben visibile, stampata SEMPRE (non condizionata a flag/verbosità).
- Chiama `animate_collapse`/`plot_summary` salvando in una sottocartella
  `output/` della root del progetto (creata se assente), stampa i
  percorsi generati.
- Output filtrato (regola CLAUDE.md): niente log grezzi di `solve_ivp`.

### 4. Test
- `tests/test_pipeline.py` (nuovo): `run_full_simulation("s15"/"s20"/"s25")`
  senza eccezioni; `KE_finale_erg >= 0`; `PE_erg < 0` con ordine di
  grandezza plausibile (vedi test di sanità sopra); classificazioni in
  `{"white_dwarf","neutron_star","black_hole"}`; `len(disclaimers) == 8`
  e ciascuna delle 8 parole chiave sopra è contenuta (case-insensitive)
  in almeno una voce di `disclaimers` (test esplicito per parola
  chiave, non un controllo generico "non vuoto").
- **(NUOVO, da review)** Test sintetico per il ramo `collapsed=False`:
  con il catalogo attuale è STRUTTURALMENTE impossibile ottenerlo dalle
  3 stelle di riferimento (`Progenitor.__post_init__` impone
  `1.3<=massa_nucleo_msun<=2.0`, sempre supra-Chandrasekhar per i Ye del
  catalogo) — il test deve quindi chiamare direttamente
  `collasso.dynamics.simulate_collapse` con un `t_max_s` volutamente
  piccolo (es. una piccola frazione del tempo di caduta libera) per
  forzare `collapsed=False`, poi verificare che il resto della pipeline
  (calcolo energie, non solo il messaggio di stampa) non sollevi
  eccezioni e gestisca l'assenza di un vero momento di collasso in modo
  coerente (non richiede necessariamente passare per `run_full_simulation`
  se questo rende il test più semplice — verificare almeno la funzione
  di calcolo energie in isolamento su un `CollapseSolution` con
  `collapsed=False`).
- `tests/test_visualization.py` (nuovo): calcola UNA `SimulationResult`
  (s20) a livello di modulo (fixture-style, riusata fra i test per non
  rallentare la suite), poi verifica che `animate_collapse`/
  `plot_summary` producano file con `os.path.getsize(...) > 0`.

### 5. Chiusura del ciclo
5.1. `pytest tests/` (intera suite) — nessuna regressione.
5.2. `python scripts/run_simulation.py s20` (non interattivo) eseguito,
     output verificato (tutti i disclaimer presenti, file GIF+PNG creati).
5.3. Passare la mano al `critic-fisico`: completezza/onestà dei 7
     disclaimer, coerenza di `relativistic=True` e SLy/APR4 con quanto
     già validato, nessuno sconfinamento.
5.4. Orchestratore: `pytest`, resoconto, aggiornamento STATUS.md.
5.5. **STOP esplicito**, attesa conferma utente prima di Step 8.

### GIORNO 7 — 2026-08-11 — Step 7 completato
**Implementato**: `collasso/pipeline.py` (nuovo) — `run_full_simulation(star_id)`,
orchestrazione pura (nessun I/O) di tutta la fisica già validata: profilo
di equilibrio (Lane-Emden, Step 2), dinamica relativistica del collasso
(`relativistic=True` passato ESPLICITAMENTE, senza toccare il default
`False` della funzione, Step 4/5), classificazione remnant con le EOS
realistiche SLy/APR4 (Retrofit Step 6, mai il path storico bacato).
Nuove energie: energia cinetica finale e autoenergia gravitazionale
discretizzata delle shell (formula a massa-a-metà-shell, corretta dopo
review per ridurre il bias sistematico di una somma di Riemann a
estremo destro), con limite dichiarato esplicito che il bilancio non è
chiuso (nessuna energia interna/termica, nessuna perdita per neutrini).
`collasso/visualization.py` (nuovo) — animazione GIF del collasso
(`FuncAnimation`, backend Agg headless) e grafici statici (densità
centrale/velocità vs tempo, curva massa-raggio TOV). `scripts/run_simulation.py`
(nuovo) — **l'entry point unico richiesto dall'utente**: chiede la
stella in input (o la accetta da riga di comando), produce insieme
output numerico completo e visualizzazione, con una sezione "LIMITI DEL
MODELLO" che elenca sempre e in modo ben visibile **8 disclaimer**
(catalogo placeholder, mismatch Lane-Emden, nessun raffreddamento
neutrinico, nessuna viscosità artificiale, correzione GR approssimata
"Case A", EOS SLy/APR4 fenomenologiche, nessuna rotazione/campo
magnetico, e — aggiunto dopo review — **simmetria sferica/1D**, segnalato
come probabilmente il limite concettualmente più importante del progetto).

**Scoperta del coder durante l'implementazione**: una premessa del piano
era sbagliata — si pensava fosse "strutturalmente impossibile" ottenere
`collapsed=False` dal catalogo di riferimento attuale, ma **s15 è
genuinamente sub-Chandrasekhar** per il suo Ye=0.50
(`chandrasekhar_mass_msun(0.50)=1.4559` > `massa_nucleo_msun=1.35`) — il
ramo `collapsed=False` si verifica quindi realmente per s15, non solo
nel test sintetico previsto dal piano. La gestione esplicita di questo
caso nell'output (mai presentare energie/densità come se un vero
collasso fosse avvenuto) è stata verificata dal critic-fisico eseguendo
`run_simulation.py s15` di persona: output corretto e onesto, s15 si
classifica "white_dwarf" con entrambe le EOS.

**Test**: `pytest tests/` (intera suite) → **91/91 passati**, 0 falliti
(81 da Step 1-6/Retrofit + 10 nuovi di Step 7). Nessuna regressione.
`python scripts/run_simulation.py s20`: collasso avvenuto,
KE_finale=2.90e51 erg, PE_iniziale=-5.76e51 erg, PE_finale=-3.90e52 erg,
entrambe SLy e APR4 classificano s20 "neutron_star", file
`output/collapse_s20.gif` (578 KB) e `output/summary_s20.png` (74 KB)
generati correttamente.

**Verifica critic-fisico**: verdetto **"conforme"** (nessuna riserva).
Ha rieseguito `run_simulation.py s15` di persona per verificare
l'onestà della gestione di `collapsed=False`; ha fatto un controllo
quantitativo indipendente sulla tolleranza allargata del test di
sanità su PE (confermando che il collasso fortemente non-omologo
concentra l'autoenergia gravitazionale a un raggio molto più piccolo
di quello della shell esterna, spiegazione fisica del coder confermata
con numeri, non un artificio per far passare il test); confermata
la formula a massa-a-metà-shell, tutti gli 8 disclaimer con le parole
chiave corrette, `relativistic=True` esplicito, uso esclusivo di
SLy/APR4 (mai il path legacy), perimetro rispettato (nessuna fisica
nuova oltre alla composizione di moduli già validati).

**Stato progetto**: 7/9 step completati.
**Prossimo step proposto**: Step 8 — Validazione qualitativa contro
GR1D. In attesa di conferma esplicita dell'utente prima di procedere.

### GIORNO 8 — 2026-08-11 — Step 8 completato
**Prodotto**: `VALIDATION.md` (nuovo, root del progetto) — confronto
qualitativo con GR1D/letteratura (O'Connor & Ott 2010/2011), come
richiesto da CLAUDE.md. Nessun'esecuzione letterale di GR1D (fuori
portata come dipendenza esterna); confronto con valori/comportamenti
pubblicati.

**Indagine prioritaria (richiesta esplicita dell'utente)**: il pattern
non monotono di velocità della shell più interna per s20 (-806 km/s a
23.8ms, risalita a -630 km/s a 50.1ms, poi -2322 km/s a 94.3ms) è stato
verificato con evidenza quantitativa dal critic-fisico: decomposizione
dei termini di gravità/pressione (si annullano quasi esattamente per
tutta la finestra 20-56ms, con la forza netta che cambia segno esatto
al massimo locale di velocità — non casuale) e test di convergenza a
parità di coordinata di massa Lagrangiana (scarto <0.1% su N=100→1600
shell, presente su tutta la stella, non confinato a poche shell di
bordo). **Verdetto: FISICO**, non un artefatto numerico — conseguenza
corretta e ben convergente del modello (s20 solo al 122% di
M_Chandrasekhar, "quasi-pausa" temporanea nella risposta di pressione
prima che l'instabilità persistente riprenda il sopravvento). **Nota
critica**: NON è il vero bounce delle supernove reali (il modello di
dinamica usa sempre l'EOS di Chandrasekhar per elettroni, che non si
irrigidisce mai a densità nucleare — nessun meccanismo di bounce
esiste in questo modello) — è un comportamento specifico di questo
modello semplificato, non una riproduzione realistica della fisica del
bounce. Segnalata (non corretta, fuori perimetro) un'osservazione
secondaria su una brusca decelerazione negli ultimi ~1.5ms, coerente
coi limiti già dichiarati (nessuna viscosità artificiale).

**Confronti standard**: tempi scala (decine-100ms, ordine di grandezza
coerente con la letteratura, con limite dichiarato che non è un vero
tempo di bounce); massa critica (riepilogo verificato di Step 3/6/
Retrofit, nessuna discrepanza); classificazione remnant (s15 white_dwarf,
s20/s25 neutron_star, qualitativamente coerente col quadro di O'Connor &
Ott 2011, con limitazione onesta dichiarata: solo 3 stelle placeholder,
nessuna supera M_TOV, impossibile testare la transizione a black_hole).

**Nessun bug trovato**: tutti i vincoli della tabella CLAUDE.md
verificati punto per punto, nessuna violazione, nessuna correzione
necessaria in questo ciclo (perimetro rispettato: validazione, non
correzione).

**Stato progetto**: 8/9 step completati.
**Prossimo step proposto**: Step 9 — Pulizia, documentazione, README,
pubblicazione GitHub. In attesa di conferma esplicita dell'utente prima
di procedere. **Nota**: CLAUDE.md richiede "nessun push pubblico su
GitHub senza conferma esplicita dell'utente" — la parte di pubblicazione

### Nota — 2026-08-13 — Verdetto "FISICO" di Step 8 riesaminato e DECLASSATO

Dopo l'integrazione del catalogo reale (ciclo "Integrazione catalogo
reale", sopra), il pattern non-monotono di velocità qui sopra descritto
per s20 non si è più ripresentato (vedi VALIDATION.md aggiornato).
L'utente ha chiesto di far verificare a `critic-fisico` un'ipotesi
precisa: il test di convergenza in N di questa voce verificava solo
l'errore di discretizzazione a parità di condizione iniziale, e non
poteva in linea di principio rivelare che il profilo iniziale
PLACEHOLDER fosse esso stesso fisicamente incoerente (n_politropico=2.0
arbitrario, NON derivato dalla EOS; mismatch di Lane-Emden fra il raggio
dichiarato e quello consistente con la massa dichiarata) — quindi il
verdetto "fisico" poteva riferirsi in realtà a un artefatto
dell'incoerenza del profilo, non della discretizzazione numerica.

**Indagine di riesame (critic-fisico, indipendente)**:
1. **Punto epistemologico confermato**: un test di convergenza in N a
   parità di condizione iniziale può SOLO rivelare errore di
   discretizzazione — la CI, per quanto incoerente, resta identica a
   ogni N, quindi la sua incoerenza è invariante in N e strutturalmente
   invisibile a questo tipo di test.
2. **Riproduzione esatta**: il vecchio profilo placeholder esatto
   (n=2.0, ρc=7.5e9, massa=1.50 Msun, ye=0.46), fatto girare sul codice
   ATTUALE di `collasso/dynamics.py` (fisica invariata), riproduce il
   pattern originale quasi identicamente (-805.8 km/s a 23.8ms, -630.2
   km/s a 50.1ms, crollo a -2322.0 km/s a 94.99ms) — conferma che il
   fenomeno era specifico di quei parametri, non un residuo di
   refactoring del codice.
3. **Esperimento isolante (decisivo)**: tenendo IDENTICI ρc=7.5e9,
   ye=0.46, massa=1.50 Msun (quindi lo STESSO rapporto massa/M_Ch=121.7%
   del vecchio caso), e cambiando SOLO n da 2.0 (arbitrario) a
   n_eff_chandrasekhar(7.5e9, 0.46)=2.97514 (EOS-consistente, come nel
   nuovo catalogo): **il pattern sparisce completamente** — zero cambi
   di segno in dv/dt su tutta la traiettoria, verificato su N=200/400/
   800/1600 e anche con ye=0.495. Il rapporto massa/M_Ch NON è la causa:
   l'unica variabile che, cambiata, elimina il fenomeno è l'incoerenza
   strutturale di n=2.0.
4. **Scarto K quantificato**: il vecchio profilo placeholder aveva
   scarto K = **+35.58%** (contro +2.3%/+6.7%/+9.5% delle stelle del
   catalogo reale, e +14.92% nell'esperimento isolante che pure aveva lo
   stesso rapporto massa/M_Ch estremo del vecchio caso ma n corretto) —
   3.7-15x più incoerente di qualunque configurazione oggi nel progetto.
   Mismatch di raggio confermato quantitativamente: 6.33x (non solo
   "~5-6x" approssimato).

**Verdetto**: **l'ipotesi dell'utente regge, con piena evidenza
quantitativa**. Il vecchio verdetto "FISICO, non artefatto numerico" va
DECLASSATO, con una precisazione: la parte NUMERICA era corretta (la
convergenza in N era vera e correttamente verificata a parità di
coordinata di massa Lagrangiana — non un errore metodologico, è
esattamente ciò che quel test poteva verificare). La parte
INTERPRETATIVA ("quindi è fisico") era un salto logico ingiustificato:
la dicotomia "fisico vs artefatto numerico" usata allora era incompleta
— ometteva una terza possibilità, qui dimostrata: **un fenomeno
numericamente convergente ma che è la conseguenza diretta e
riproducibile di una condizione iniziale essa stessa incoerente**, non
una caratteristica fisica del modello di collasso. Il pattern non era né
"il vero bounce" (già escluso correttamente in Step 8) né una feature
fisica minore del modello semplificato (come invece concluso allora) —
era un artefatto del profilo iniziale placeholder, mascherato da una
convergenza numerica che, per costruzione, non poteva rivelarlo.

**Lezione di metodo per il progetto**: un test di convergenza in N
verifica la fedeltà della discretizzazione alla condizione iniziale
data, non la validità fisica di quella condizione iniziale — le due
cose vanno sempre verificate separatamente (la seconda richiede un
controllo di autoconsistenza del profilo stesso, come lo "scarto K"
introdotto nel ciclo successivo, non un test di convergenza).
di Step 9 andrà confermata separatamente anche se lo step viene
avviato.

## Ciclo "Integrazione catalogo reale" — 2026-08-13 — completato

Ciclo separato, precedente a Step 9, richiesto esplicitamente
dall'utente: collegare il catalogo reale dei progenitori (al posto dei 3
placeholder) come pezzo sostanziale, non cosmetico.

**Vincolo esplicito dell'utente** (rifiutato un parser automatico su
2sn.org, formato non documentato): cercare in letteratura un paper con
una TABELLA (non un dataset grezzo) di parametri pre-collasso per 5-10
stelle standard, preferibilmente famiglia s15/s20/s25, trascrivere a
mano con citazione precisa, riportare le fonti trovate PRIMA di
implementare.

**Ricerca svolta**: controllati Sukhbold, Woosley & Heger (2016, HTML
non renderizzato), Woosley Heger & Weaver (2002, nessun arXiv ID
pulito), Sukhbold & Woosley (2014, ApJ, "The Compactness of Presupernova
Stellar Cores" — Tabella 2 pag. 21, masse nuclei He/CO per 15/20/25
Msun, utile riscontro ma non il dato primario scelto), Heger, Fryer,
Woosley, Langer & Hartmann (2003, ApJ 591:288 — schema di classificazione
popolazionale, nessuna tabella per-stella coi parametri richiesti), e
**O'Connor & Ott (2011)**, "Black Hole Formation in Failing Core-Collapse
Supernovae", ApJ 730, 70 (arXiv:1010.5550) — **Tabella 1 pag. 5, scelta
come fonte primaria**: contiene esattamente i modelli s15WHW02/
s20WHW02/s25WHW02 (metallicità solare, dati originali di Woosley, Heger
& Weaver 2002 — la stessa fonte della convenzione di naming già in uso),
con massa del nucleo di ferro (definita come massa baryonica interna a
Ye=0.495) e ξ2.5. Fonti riportate all'utente e approccio confermato
prima di implementare.

**Limite onesto riscontrato e riportato**: nessuno dei paper controllati
tabula densità centrale o raggio iniziale pre-collasso per stella
nominata (solo massa del nucleo e, in O'Connor&Ott, un parametro di
compattezza ξ2.5 definito al bounce, non pre-collasso). Approccio
concordato con l'utente: massa_nucleo_msun e ye da letteratura reale
(citati); densita_centrale_gcm3 scelta come valore fisico plausibile
(NON letteratura, dichiarato esplicitamente); n_politropico e
raggio_iniziale_km DERIVATI internamente dal progetto stesso
(collasso.eos.n_eff_chandrasekhar + collasso.lane_emden), non presi da
letteratura — provenienza per campo documentata esplicitamente in
`data/progenitors_reference.csv`, `collasso/catalog.py` e nei disclaimer
di `collasso/pipeline.py`.

**Verifica richiesta esplicitamente dall'utente — classificazione s15**:
con ye=0.495, M_Ch=1.4270 Msun (invariato, dipende solo da ye). Le tre
masse reali (s15=1.55, s20=1.46, s25=1.62 Msun) sono TUTTE sopra M_Ch
(108.6%/102.3%/113.5%), contro il vecchio catalogo placeholder dove solo
s15 era SOTTO M_Ch (1.35/1.4559=92.7%). **Risultato confermato**: s15
passa da `white_dwarf` a `neutron_star` (sia con EOS SLy sia APR4) — la
massa vera del nucleo di ferro (superiore al placeholder) rende s15
correttamente instabile, come atteso per un vero progenitore da 15 Msun
ZAMS. s20/s25 restano `neutron_star` con entrambe le EOS (nessuna
stella supera M_TOV~2.0-2.2 Msun, nessuna diventa black_hole).

**Scoperta fisica aggiuntiva** (non anticipata, verificata
quantitativamente): con ye=0.495 nessuna delle tre masse reali del
nucleo ammette un vero equilibrio idrostatico con la EOS di Chandrasekhar
esatta (la massa di un tale equilibrio, per costruzione, è sempre <
M_Ch) — atteso e corretto: sono nuclei supra-Chandrasekhar per
definizione, è la ragione per cui collassano. `n_politropico` e
`raggio_iniziale_km` vanno quindi intesi come condizione iniziale
approssimata dell'istante appena prima del collasso, non equilibrio
vero — dichiarato esplicitamente nel disclaimer aggiornato #2.

**Conseguenza numerica scoperta e corretta**: con il nuovo profilo
auto-consistente, s20 (il caso più marginalmente supra-Chandrasekhar,
102.3% di M_Ch) richiede un tempo di integrazione maggiore per
raggiungere la soglia di collasso (~12.6x il tempo di caduta libera in
Newtoniano, ~11x con la correzione relativistica, contro ~10x per
s15/s25). `collasso.dynamics.T_MAX_FREE_FALL_MULTIPLIER` alzato da 10.0
a 15.0 (verificato con margine su tutte e tre le stelle, entrambe le
modalità relativistic=True/False) — pura estensione del budget di
integrazione numerica, nessun nuovo vincolo fisico introdotto, verificato
da critic-fisico.

**File modificati**: `data/progenitors_reference.csv` (dati reali),
`collasso/catalog.py` (docstring aggiornate), `collasso/pipeline.py`
(disclaimer #1/#2 aggiornati), `collasso/dynamics.py`
(T_MAX_FREE_FALL_MULTIPLIER), `tests/test_catalog.py`,
`tests/test_pipeline.py` (keyword disclaimer aggiornate),
`tests/test_relativistic.py` (baseline di regressione per s20
ri-catturata sui nuovi dati — dynamics.py stesso non modificato nella
sua fisica). Suite completa: 91 test, tutti passano.

**Nota per lo step successivo**: `VALIDATION.md` (Step 8) contiene
numeri riferiti al VECCHIO catalogo placeholder (inclusa la tabella di
classificazione remnant con s15=white_dwarf, ora obsoleta) — da
aggiornare in un ciclo successivo, non in questo (fuori perimetro
esplicito di questo ciclo, che l'utente ha delimitato a
s15/s20/s25 senza espandere ad altre stelle).

**Stato progetto**: catalogo reale integrato per s15/s20/s25 (non le
9+ stelle del catalogo completo Sukhbold/Woosley/Heger — limitazione
dichiarata, coerente col vincolo dell'utente di 5-10 stelle trascritte a
mano). Step 9 resta il prossimo step proposto, in attesa di conferma
esplicita dell'utente.

## Ciclo "Chiusura autoconsistenza densità + VALIDATION.md" — 2026-08-13

Sub-ciclo immediato, richiesto dall'utente prima di Step 9, in risposta a
due domande dirette sul ciclo precedente.

**Domanda 1 — come è stata scelta la densità centrale?** Risposta
verificata: era un valore LIBERO (stesso ordine di grandezza del vecchio
placeholder), non uno shooting — `physical_profile` fa tornare la massa
per costruzione algebrica (inversione della scala radiale), non è una
verifica di equilibrio fisico. Investigato se uno shooting genuino (legare
anche la costante di normalizzazione K alla EOS esatta, non solo la
pendenza locale n) potesse chiudere l'autoconsistenza: **dimostrato che è
matematicamente impossibile** per queste tre stelle — con la EOS esatta di
Chandrasekhar, la massa di un vero equilibrio idrostatico M(ρc) cresce
monotonamente con ρc ma non supera mai M_Ch, a nessuna densità (verificato
fino a ρc=1e20 g/cm³, non fisico: M(ρc)/M_Ch resta <100.0000%). Tutte e
tre le masse reali (1.55/1.46/1.62 Msun) sono sopra M_Ch=1.4270 Msun,
quindi nessun equilibrio esiste per nessuna di esse — è la ragione fisica
diretta per cui collassano, non un limite del metodo.

**Decisione utente**: mantenere l'approccio attuale (densità libera +
massa esatta per costruzione), ma esporre esplicitamente lo scarto
residuo (chiamato "scarto K") nell'output finale e nel disclaimer per
ogni stella, non solo nella documentazione interna, specificando che è la
misura diretta e attesa di quanto ciascun nucleo sia supra-Chandrasekhar,
non un errore di modellazione.

**Implementato**: nuova funzione `collasso.eos.chandrasekhar_k_deviation_pct`
(confronta la costante politropica implicita nel profilo con quella reale
della EOS di Chandrasekhar alla stessa densità/n); nuovo campo
`SimulationResult.k_deviation_pct` in `collasso/pipeline.py`; disclaimer
#2 ora interpola il valore numerico per la stella in corso
(`_build_disclaimers` ora prende `k_deviation_pct` come parametro);
`scripts/run_simulation.py` stampa esplicitamente lo scarto K nella
sezione "Profilo di equilibrio iniziale"; corretto anche un commento
obsoleto in quello script (lo scarto raggio_iniziale_km vs R_derivato_km
ora è ~0% by construction, non più il vecchio mismatch placeholder).
2 nuovi test in `tests/test_eos.py` (scarto nullo per un profilo
genuinamente EOS-consistente; scarto positivo e nell'ordine di grandezza
atteso per una massa supra-Chandrasekhar). Valori confermati: s15=+6.730%,
s20=+2.320%, s25=+9.521% — ordinamento coerente con quanto ciascuna
stella è sopra M_Ch (s20 il meno supra-Ch ha lo scarto minore, s25 il più
supra-Ch ha lo scarto maggiore).

**Domanda 2 — aggiornare subito VALIDATION.md**: fatto. `critic-fisico`
ha rifatto l'intera indagine di Step 8 sul codice/catalogo attuale
(indipendentemente, non solo verificando i numeri dell'orchestratore).
**Scoperta rilevante**: il pattern non-monotono di velocità per s20
(risultato principale della vecchia versione di Step 8) **non è più
presente** con la massa reale del nucleo — il nuovo profilo (n≈2.976,
molto più vicino a 3) produce un piccolo transiente iniziale verso
l'esterno (0-4ms, fino a +709 km/s), poi un infall liscio e monotono per
l'80% della traiettoria, poi oscillazioni numeriche tardive di ampiezza
DECRESCENTE con N (N=200→800: da -1517 a -1013 km/s) — quindi un
artefatto numerico noto (assenza di viscosità artificiale), non una nuova
feature fisica. `VALIDATION.md` riscritto interamente con questi numeri
(tempi scala, massa critica, classificazione, nuova sezione dedicata allo
scarto K, tabella vincoli) — nessun numero residuo del vecchio catalogo
placeholder.

**File modificati**: `collasso/eos.py` (nuova funzione),
`collasso/pipeline.py` (nuovo campo, disclaimer parametrizzato),
`scripts/run_simulation.py` (stampa scarto K, fix commento obsoleto),
`tests/test_eos.py` (2 nuovi test), `VALIDATION.md` (riscritto). Suite
completa: 93 test, tutti passano.

**Stato progetto**: nessun documento residuo con numeri del vecchio
catalogo placeholder. Step 9 resta il prossimo step proposto, in attesa
di conferma esplicita dell'utente.

## Ciclo "Riesame verdetto Step 8" — 2026-08-13 — completato

Sub-ciclo immediato, richiesto dall'utente prima di Step 9: verificare se
il vecchio verdetto "FISICO" di Step 8 sul pattern di velocità di s20
(sparito col catalogo reale, vedi ciclo precedente) fosse in realtà un
artefatto dell'incoerenza del profilo placeholder, non rivelabile dal
test di convergenza in N allora eseguito.

**Esito**: ipotesi confermata con piena evidenza quantitativa da
`critic-fisico` (indagine indipendente, dettaglio completo nella nota
"Verdetto 'FISICO' di Step 8 riesaminato e DECLASSATO" inserita subito
dopo il log originale di GIORNO 8, sopra). In sintesi: l'esperimento
isolante (stesso rapporto massa/M_Ch=121.7% del vecchio caso, solo n
cambiato da 2.0 arbitrario a n_eff_chandrasekhar EOS-consistente) fa
sparire completamente il pattern — prova diretta che la causa era
l'incoerenza strutturale del profilo (scarto K +35.58%, mismatch di
raggio 6.33x), non il rapporto massa/M_Ch. Il vecchio verdetto "FISICO"
è stato declassato: la convergenza numerica era vera, ma la conclusione
"quindi fisico" era un salto logico ingiustificato — mancava la terza
possibilità (fenomeno convergente ma causato da una condizione iniziale
incoerente), ora esplicitamente documentata come lezione di metodo.

**File modificati**: `STATUS.md` (nota di correzione dopo GIORNO 8),
`VALIDATION.md` (nota storica §1 e Conclusione riscritte per riflettere
il verdetto declassato). Nessuna modifica al codice (indagine forense
pura, nessun bug trovato nel codice attuale — il vecchio pattern era
un fatto sul VECCHIO catalogo placeholder, non sul codice).

**Stato progetto**: nessun documento del progetto attribuisce più
erroneamente un significato fisico al vecchio pattern di velocità
placeholder. Step 9 resta il prossimo step proposto, in attesa di
conferma esplicita dell'utente.

## Ciclo "Step 9 (parte 1: pulizia + README)" — 2026-08-14

Eseguito in plan mode (piano approvato dall'utente). Perimetro: SOLO
pulizia del codice e documentazione — nessun `git init`/commit/push
(regola CLAUDE.md, richiesta esplicita dell'utente di vedere prima il
README).

**Doppio controllo di validità/consistenza** (richiesto esplicitamente
dall'utente prima di Step 9): riletti per intero, riga per riga, tutti i
12 moduli di `collasso/` (inclusi quelli non toccati nei cicli recenti:
`constants.py`, `relativistic.py`, `remnant.py`, `tov.py`,
`eos_piecewise_polytrope.py`, `eos_neutron.py`, `visualization.py`,
`dynamics.py` per intero) — tutto internamente coerente, tutte le
correzioni storiche (bug fisico TOV, formule relativistiche, EOS
realistica) ancora correttamente applicate, nessuna regressione trovata.

**Pulizia**:
- `requirements.txt`: rimossa `pandas` (confermato non importata da
  nessuna parte nel codice sorgente — grep pulito su `collasso/`,
  `scripts/`, `tests/`, `conftest.py`).
- Scansione statica con `pyflakes` su `collasso/` e `scripts/`: zero
  problemi trovati (import inutilizzati, variabili morte) — conferma la
  scansione manuale già fatta.
- `.gitignore` (nuovo): esclude `.venv/`, `__pycache__/`, `.pytest_cache/`,
  `output/` (rigenerabile).
- `docs/img/summary_s20.png` (nuovo): copia statica di un'immagine di
  esempio, per il README, senza tracciare l'intera cartella `output/`
  rigenerabile.

**README.md** (nuovo, root del progetto): obiettivo del progetto, tabella
dei vincoli fisici con approfondimento esplicito sui due punti richiesti
dall'utente (assenza totale — non "semplificata" — del trasporto
neutrinico; assenza di un vero bounce fisico, per costruzione dell'EOS di
Chandrasekhar), struttura del repository, installazione, comando esatto
per `run_simulation.py` con esempio di output reale, tabella degli script
demo storici, sezione test, riassunto di VALIDATION.md, e una sezione
dedicata "Rigore metodologico: come il progetto ha trattato i propri
errori" che racconta per esteso la correzione del verdetto di Step 8
(richiesta esplicita dell'utente: "è un punto di forza da mostrare, non
da nascondere"). Licenza dichiarata esplicitamente come NON ancora
decisa, non inventata di iniziativa.

**Verifica finale**: suite completa rieseguita dopo la rimozione di
pandas — 93/93 test passano; `python scripts/run_simulation.py s20`
rieseguito con successo (nessun ImportError, GIF/PNG rigenerati
correttamente); README riletto e confrontato numero per numero contro
codice/file reali (nessun valore inventato).

**File modificati/creati**: `requirements.txt`, `.gitignore` (nuovo),
`docs/img/summary_s20.png` (nuovo), `README.md` (nuovo).

**Stato progetto**: pulizia e documentazione di Step 9 completate.
Prossimo passaggio: mostrare il README all'utente, poi attendere la sua
decisione su licenza e su `git init`/pubblicazione GitHub — nessun push
senza conferma esplicita (CLAUDE.md).

## Ciclo "Espansione catalogo + Betelgeuse" — 2026-08-14

Ultimo ciclo prima della chiusura di Step 9, richiesto esplicitamente
dall'utente con quattro parti: (1) espandere il catalogo con s30WHW02/
s35WHW02/s40WHW02 dalla stessa Tabella 1 di O'Connor & Ott 2011, stesso
metodo gia' validato per s15/s20/s25; (2) aggiornare il README col
catalogo completo; (3) confermare/aggiungere la visibilita' dello scarto K
nell'output completo; (4) licenza MIT; (5, aggiunta): un proxy dichiarato
per Betelgeuse, con etichettatura onesta.

**Espansione catalogo (s30/s35/s40)**: stesso metodo gia' approvato per
s15/s20/s25 — massa nucleo e ye da O'Connor & Ott 2011 Tabella 1
(s30WHW02: 1.46 Msun; s35WHW02: 1.49 Msun; s40WHW02: 1.56 Msun; ye=0.495
per tutte), densita' centrale scelta come valore fisico plausibile
(prosegue la progressione aritmetica gia' usata: 1.25e10/1.50e10/1.75e10
g/cm^3), n_politropico e raggio_iniziale_km derivati internamente
(n_eff_chandrasekhar + Lane-Emden). Scarto K per-stella: s30=+2.10%,
s35=+3.43%, s40=+6.59% — coerente con quanto ogni nucleo e' sopra
M_Ch=1.4270 Msun (102.3%/104.4%/109.3%).

**Verifica scarto K nell'output**: confermato che `k_deviation_pct` era
GIA' presente nell'output completo di `scripts/run_simulation.py`
(sezione "Profilo di equilibrio iniziale", aggiunto nel ciclo "Chiusura
autoconsistenza densita'") — l'estratto mostrato nel README era solo
troncato per brevita'. README aggiornato per includere esplicitamente
questa riga nell'estratto, cosi' la sua presenza sia visibile senza dover
rieseguire lo script.

**Proxy Betelgeuse**: richiesta esplicita dell'utente di cercare, nella
griglia WHW02 gia' in uso (Tabella 1 di O'Connor & Ott 2011: 7 masse ZAMS
disponibili per metallicita' solare, 15/20/25/30/35/40/75 — nessuna massa
intermedia come 17 o 18), il modello di massa iniziale piu' vicina alla
stima reale di Betelgeuse, verificando la fonte piu' solida (non a
memoria). Ricerca effettuata (WebSearch/WebFetch): Joyce, Leung, Molnar,
Ireland, Kobayashi & Nomoto (2020), ApJ 902, 63 — "we report model-derived
estimates for the initial and present-day masses of Betelgeuse as
approximately 18-21 Msun and 16.5-19 Msun respectively" (il valore
16.5-19 Msun spesso citato in letteratura divulgativa e' la massa
ATTUALE, non quella ZAMS — distinzione verificata leggendo l'abstract
completo, non assunta). Confermato indipendentemente da Saio, Kondo,
Ekstrom, Meynet & Georgy (2023), MNRAS 526, 2765 (arXiv:2306.00287):
modelli con proprieta' di pulsazione compatibili con Betelgeuse hanno
massa attuale 11-12 Msun, 19 Msun a ZAMS. Entrambe le fonti convergono su
ZAMS~18-21 Msun: **s20WHW02 (ZAMS=20) e' il valore piu' vicino disponibile
in Tabella 1**, con margine chiaro su s15 (ZAMS=15) o s25 (ZAMS=25).

Implementato come nuova voce del catalogo, `betelgeuse`, che riusa
ESATTAMENTE i valori numerici di `s20` (massa nucleo, ye, densita'
centrale, n_politropico, raggio_iniziale_km — stesso modello sottostante),
distinta solo per `id` e per il campo `fonte`/nuovo campo `nota_proxy`,
che etichetta esplicitamente: "proxy: modello generico WHW02 di massa
piu' vicina; NON una ricostruzione della struttura interna specifica di
Betelgeuse". Per rendere questa etichettatura impossibile da perdere
(richiesta esplicita: "sia nel CSV sia nei disclaimer di output"),
`collasso.pipeline._build_disclaimers` ora accetta `nota_proxy` e aggiunge
un NONO disclaimer condizionale (solo per voci con `nota_proxy` non
vuota) — verificato con un'esecuzione reale (`python scripts/
run_simulation.py betelgeuse`) che il disclaimer 9 compare correttamente.

**File modificati**: `collasso/catalog.py` (nuovo campo `Progenitor.
nota_proxy`, opzionale, default vuoto), `data/progenitors_reference.csv`
(3 nuove righe reali + 1 riga proxy, nuova colonna `nota_proxy`),
`collasso/pipeline.py` (`_build_disclaimers` accetta `nota_proxy`,
disclaimer 1 aggiornato per 6 stelle, nono disclaimer condizionale),
`tests/test_catalog.py` (7 voci attese, nuovi test su `nota_proxy` e sul
riuso esatto dei valori di s20 per betelgeuse), `tests/test_pipeline.py`
(STAR_IDS esteso a 6 stelle, nuovo test dedicato per betelgeuse — 9
disclaimer + parola chiave "proxy"), `README.md` (nuova sezione "Catalogo
dei progenitori" con tabella completa a 7 righe e spiegazione del proxy,
esempio di output esteso con la riga dello scarto K, sezione Licenza
aggiornata), `LICENSE` (nuovo, MIT — placeholder `[nome del titolare del
copyright]` da sostituire, non inventato).

**Verifica finale**: `pyflakes` su `collasso`/`scripts` pulito; suite
completa rieseguita — **102/102 test passano** (93 + 9: 6 istanze in piu'
dai due test parametrizzati su STAR_IDS esteso, 3 nuovi test dedicati a
betelgeuse); `python scripts/run_simulation.py betelgeuse` rieseguito con
successo, disclaimer 9 verificato presente nell'output reale.

**Stato progetto**: catalogo esteso a 6 stelle reali + 1 proxy dichiarato,
README aggiornato, licenza MIT aggiunta (nome del titolare da inserire).
Step 9 pronto per la chiusura, in attesa della decisione dell'utente su
`git init`/pubblicazione GitHub — nessun push senza conferma esplicita
(CLAUDE.md).
</content>
